import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 48},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16},
            num_warps=2,
            num_stages=1,
        ),
    ],
    key=["N", "C_OUT", "H_OUT", "W_OUT"],
)
@triton.jit
def conv_transpose2d_kernel(
    x_ptr,          # *[N, C_IN, H_IN, W_IN]
    w_ptr,          # *[C_IN, C_OUT, K_H, K_W]
    bias_ptr,       # *[C_OUT] or dummy
    y_ptr,          # *[N, C_OUT, H_OUT, W_OUT]
    N, C_IN, H_IN, W_IN,
    C_OUT,
    H_OUT, W_OUT,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Matrix dimensions
    M = N * H_OUT * W_OUT

    # Offsets along output matrix M (rows) and C_OUT (cols)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_OUT

    # Map row indices -> (n, oh, ow)
    hw = H_OUT * W_OUT
    n_idx = offs_m // hw
    tmp = offs_m - n_idx * hw
    oh_idx = tmp // W_OUT
    ow_idx = tmp - oh_idx * W_OUT

    # Precompute padded output coordinates
    oh_plus_pad = oh_idx + PADDING_H
    ow_plus_pad = ow_idx + PADDING_W

    # Broadcasted indices that do not depend on K loop
    n_b = n_idx[:, None]                     # [BM, 1]
    oh_base = oh_plus_pad[:, None]          # [BM, 1]
    ow_base = ow_plus_pad[:, None]          # [BM, 1]
    oc_b = offs_n[None, :]                  # [1, BN]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction dimension size
    K = C_IN * K_H * K_W
    khw = K_H * K_W
    k_range = tl.arange(0, BLOCK_K)

    # Main K loop
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + k_range          # [BK]
        mask_k = offs_k < K

        # Map flattened K index -> (ic, kh, kw)
        ic_idx = offs_k // khw
        tmp_k = offs_k - ic_idx * khw
        kh_idx = tmp_k // K_W
        kw_idx = tmp_k - kh_idx * K_W

        # Broadcast along M / N dimensions
        kh_b = kh_idx[None, :]              # [1, BK]
        kw_b = kw_idx[None, :]              # [1, BK]

        # Compute corresponding input spatial indices
        tmp_h = oh_base - kh_b              # [BM, BK]
        tmp_w = ow_base - kw_b              # [BM, BK]

        ih = tmp_h // STRIDE_H
        iw = tmp_w // STRIDE_W

        # Validity conditions for input coordinates
        valid_h = (
            (tmp_h >= 0)
            & (tmp_h < H_IN * STRIDE_H)
            & (tmp_h % STRIDE_H == 0)
        )
        valid_w = (
            (tmp_w >= 0)
            & (tmp_w < W_IN * STRIDE_W)
            & (tmp_w % STRIDE_W == 0)
        )
        cond_hw = valid_h & valid_w

        # Broadcasted input-channel indices
        ic_b = ic_idx[None, :]              # [1, BK]

        # Final mask for loading from x
        mask_a = cond_hw & mask_m[:, None] & mask_k[None, :]

        # Linear index into x: [N, C_IN, H_IN, W_IN]
        x_index = (((n_b * C_IN + ic_b) * H_IN + ih) * W_IN + iw)

        a = tl.load(x_ptr + x_index, mask=mask_a, other=0.0)

        # Load weight tile: w layout [C_IN, C_OUT, K_H, K_W]
        ic_k = ic_idx[:, None]              # [BK, 1]
        kh_k = kh_idx[:, None]              # [BK, 1]
        kw_k = kw_idx[:, None]              # [BK, 1]

        w_index = (((ic_k * C_OUT + oc_b) * K_H + kh_k) * K_W + kw_k)
        mask_b = mask_k[:, None] & mask_n[None, :]

        b = tl.load(w_ptr + w_index, mask=mask_b, other=0.0)

        # Matrix multiply-accumulate
        acc += tl.dot(a, b)

    # Bias add (if enabled)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    # Store result back to y: [N, C_OUT, H_OUT, W_OUT]
    n_out = n_idx[:, None]
    oh_out = oh_idx[:, None]
    ow_out = ow_idx[:, None]

    y_index = (((n_out * C_OUT + oc_b) * H_OUT + oh_out) * W_OUT + ow_out)
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
    # Fallback for unsupported configs
    if groups != 1:
        return torch.nn.functional.conv_transpose2d(
            x, weight, bias, stride=stride, padding=padding,
            output_padding=output_padding, groups=groups
        )

    x = x.contiguous()
    w = weight.contiguous()

    # Shapes
    N, C_IN, H_IN, W_IN = x.shape
    C_IN_w, C_OUT_per_group, K_H, K_W = w.shape
    assert C_IN_w == C_IN, "Input channels must match weight shape for groups=1"
    C_OUT = C_OUT_per_group  # groups == 1

    # Normalize stride / padding / output_padding
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

    # No dilation (same as nn.ConvTranspose2d default)
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

    # Bias handling
    has_bias = bias is not None
    if has_bias:
        bias_contig = bias.contiguous()
    else:
        bias_contig = torch.empty(1, device=x.device, dtype=x.dtype)

    # Matrix dims
    M = N * H_OUT * W_OUT

    # Grid: program IDs over (M, C_OUT)
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(C_OUT, meta["BLOCK_N"]),
        )

    conv_transpose2d_kernel[grid](
        x, w, bias_contig, y,
        N, C_IN, H_IN, W_IN,
        C_OUT,
        H_OUT, W_OUT,
        K_H, K_W,
        stride_h, stride_w,
        pad_h, pad_w,
        HAS_BIAS=has_bias,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated ConvTranspose2d module using an optimized Triton kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        output_padding: int | tuple = 0,
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
