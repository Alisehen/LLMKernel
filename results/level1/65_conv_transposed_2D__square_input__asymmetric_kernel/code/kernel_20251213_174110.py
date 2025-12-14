import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,          # float*  [N, C_IN, H_IN, W_IN]
    w_ptr,          # float*  [K, C_OUT] with K = C_IN * K_H * K_W
    bias_ptr,       # float*  [C_OUT]
    y_ptr,          # float*  [N, C_OUT, H_OUT, W_OUT]
    N, C_IN, H_IN, W_IN,
    C_OUT,
    H_OUT, W_OUT,
    K_H, K_W,
    STRIDE_H, STRIDE_W,
    PADDING_H, PADDING_W,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Output matrix dimensions:
    #   M = N * H_OUT * W_OUT   (rows)
    #   N = C_OUT               (cols)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    M = N * H_OUT * W_OUT
    # Offsets in output matrix
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_OUT

    # Map row indices -> (n, oh, ow)
    tmp_m = offs_m
    hw = H_OUT * W_OUT
    n_idx = tmp_m // hw
    tmp_m = tmp_m % hw
    oh_idx = tmp_m // W_OUT
    ow_idx = tmp_m % W_OUT

    # Prepare accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K = C_IN * K_H * K_W

    # Loop over K dimension tiles
    k_iter = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + k_iter
        mask_k = offs_k < K

        # Map flattened K index -> (ic, kh, kw)
        khw = K_H * K_W
        ic_idx = offs_k // khw
        tmp_k = offs_k % khw
        kh_idx = tmp_k // K_W
        kw_idx = tmp_k % K_W

        # Compute corresponding input spatial indices (ih, iw)
        # oh = ih * STRIDE_H - PADDING_H + kh  => ih = (oh + PADDING_H - kh) / STRIDE_H
        # ow = iw * STRIDE_W - PADDING_W + kw  => iw = (ow + PADDING_W - kw) / STRIDE_W
        oh_b = oh_idx[:, None]
        ow_b = ow_idx[:, None]
        kh_b = kh_idx[None, :]
        kw_b = kw_idx[None, :]

        tmp_h = oh_b + PADDING_H - kh_b
        tmp_w = ow_b + PADDING_W - kw_b

        # Integer division/mod for stride condition
        # valid if tmp >= 0, tmp % STRIDE == 0, and resulting index in-range
        ih = tmp_h // STRIDE_H
        iw = tmp_w // STRIDE_W

        cond_h = (tmp_h >= 0) & (tmp_h % STRIDE_H == 0) & (ih >= 0) & (ih < H_IN)
        cond_w = (tmp_w >= 0) & (tmp_w % STRIDE_W == 0) & (iw >= 0) & (iw < W_IN)

        # Broadcast n_idx and ic_idx
        n_b = n_idx[:, None]
        ic_b = ic_idx[None, :]

        # Final mask for A tile (input)
        mask_a = cond_h & cond_w & mask_m[:, None] & mask_k[None, :]

        # Compute linear indices into x: [N, C_IN, H_IN, W_IN]
        x_index = (((n_b * C_IN + ic_b) * H_IN + ih) * W_IN + iw)
        x_index = x_index.to(tl.int64)

        a = tl.load(x_ptr + x_index, mask=mask_a, other=0.0)

        # Load B tile from w_ptr: [K, C_OUT] row-major
        w_index = offs_k[:, None] * C_OUT + offs_n[None, :]
        w_index = w_index.to(tl.int64)
        mask_b = mask_k[:, None] & mask_n[None, :]
        b = tl.load(w_ptr + w_index, mask=mask_b, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Add bias if needed
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    # Store result back to y: [N, C_OUT, H_OUT, W_OUT]
    n_b = n_idx[:, None]
    oh_b = oh_idx[:, None]
    ow_b = ow_idx[:, None]

    y_index = (((n_b * C_OUT + offs_n[None, :]) * H_OUT + oh_b) * W_OUT + ow_b)
    y_index = y_index.to(tl.int64)
    mask_y = mask_m[:, None] & mask_n[None, :]

    tl.store(y_ptr + y_index, acc, mask=mask_y)


def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    output_padding,
    groups: int = 1,
) -> torch.Tensor:
    # Fallback for unsupported configs (notably groups > 1)
    if groups != 1:
        return F.conv_transpose2d(
            x, weight, bias, stride=stride, padding=padding,
            output_padding=output_padding, groups=groups
        )

    x = x.contiguous()
    w = weight.contiguous()

    # Extract shapes
    N, C_IN, H_IN, W_IN = x.shape
    C_IN_w, C_OUT_per_group, K_H, K_W = w.shape
    assert C_IN_w == C_IN, "Input channels must match weight shape for groups=1"
    C_OUT = C_OUT_per_group  # groups == 1

    # Handle stride / padding / output_padding as tuples
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    if isinstance(output_padding, int):
        out_pad_h = out_pad_w = output_padding
    else:
        out_pad_h, out_pad_w = output_padding

    # No dilation support in this kernel; nn.ConvTranspose2d default is 1
    dilation_h = 1
    dilation_w = 1

    # Output spatial size (PyTorch formula)
    H_OUT = (H_IN - 1) * stride_h - 2 * pad_h + dilation_h * (K_H - 1) + out_pad_h + 1
    W_OUT = (W_IN - 1) * stride_w - 2 * pad_w + dilation_w * (K_W - 1) + out_pad_w + 1

    # Allocate output
    y = torch.empty(
        (N, C_OUT, H_OUT, W_OUT),
        device=x.device,
        dtype=x.dtype,
    )

    # Flatten weights to [K, C_OUT]
    K = C_IN * K_H * K_W
    w_mat = w.view(K, C_OUT).contiguous()

    # Bias handling
    has_bias = bias is not None
    if has_bias:
        bias_contig = bias.contiguous()
    else:
        # Dummy tensor (unused when has_bias is False)
        bias_contig = torch.empty(1, device=x.device, dtype=x.dtype)

    # Launch configuration
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    M = N * H_OUT * W_OUT

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(C_OUT, meta["BLOCK_N"]),
    )

    conv_transpose2d_kernel[grid](
        x, w_mat, bias_contig, y,
        N, C_IN, H_IN, W_IN,
        C_OUT,
        H_OUT, W_OUT,
        K_H, K_W,
        stride_h, stride_w,
        pad_h, pad_w,
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated version of the given ConvTranspose2d model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias
        stride = self.conv_transpose2d.stride
        padding = self.conv_transpose2d.padding
        output_padding = self.conv_transpose2d.output_padding
        groups = self.conv_transpose2d.groups

        return triton_conv_transpose2d(
            x, w, b, stride=stride, padding=padding,
            output_padding=output_padding, groups=groups
        )
