import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv2d_fwd_kernel(
    x_ptr,        # *fptr,  [N, Ci, H, W]
    w_ptr,        # *fptr,  [Co, Ci, Kh, Kw]
    b_ptr,        # *fptr,  [Co] (can be dummy if has_bias=0)
    out_ptr,      # *fptr,  [N, Co, H_out, W_out]
    N,            # int
    Ci,           # int
    H,            # int
    W,            # int
    Co,           # int
    H_out,        # int
    W_out,        # int
    Kh,           # int
    Kw,           # int
    stride_h,     # int (assumed same for width)
    padding_h,    # int (assumed same for width)
    dilation_h,   # int (assumed same for width)
    K,            # int = Ci * Kh * Kw
    M,            # int = N * H_out * W_out
    has_bias,     # int (0 or 1)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(axis=0)  # tile in output "M" dimension (N*H_out*W_out)
    pid_n = tl.program_id(axis=1)  # tile in output channel dimension (Co)

    # Offsets in M and N dimensions for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < Co

    # Compute (n_idx, h_out_idx, w_out_idx) for each row m
    HWO = H_out * W_out
    n_idx = offs_m // HWO
    tmp_m = offs_m % HWO
    h_out_idx = tmp_m // W_out
    w_out_idx = tmp_m % W_out

    # Input & weight strides (assuming contiguous tensors)
    in_n_stride = Ci * H * W
    in_c_stride = H * W
    in_h_stride = W
    in_w_stride = 1

    w_co_stride = Ci * Kh * Kw
    w_ci_stride = Kh * Kw
    w_kh_stride = Kw
    w_kw_stride = 1

    out_n_stride = Co * H_out * W_out
    out_c_stride = H_out * W_out
    out_h_stride = W_out
    out_w_stride = 1

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop (Ci * Kh * Kw)
    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Decompose K index into (ci, kh, kw)
        ci_idx = offs_k // (Kh * Kw)
        tmp_k = offs_k % (Kh * Kw)
        kh_idx = tmp_k // Kw
        kw_idx = tmp_k % Kw

        # ---- Load weight tile [BLOCK_K, BLOCK_N] ----
        # Offsets for weight: [Co, Ci, Kh, Kw]
        w_offsets = (
            offs_n[None, :] * w_co_stride
            + ci_idx[:, None] * w_ci_stride
            + kh_idx[:, None] * w_kh_stride
            + kw_idx[:, None] * w_kw_stride
        )
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # ---- Load input tile [BLOCK_M, BLOCK_K] ----
        # Compute input spatial coords for each (m,k)
        h_in = (
            h_out_idx[:, None] * stride_h
            - padding_h
            + kh_idx[None, :] * dilation_h
        )
        w_in = (
            w_out_idx[:, None] * stride_h
            - padding_h
            + kw_idx[None, :] * dilation_h
        )

        mask_h = (h_in >= 0) & (h_in < H)
        mask_w = (w_in >= 0) & (w_in < W)
        in_bounds = mask_h & mask_w

        x_offsets = (
            n_idx[:, None] * in_n_stride
            + ci_idx[None, :] * in_c_stride
            + h_in * in_h_stride
            + w_in * in_w_stride
        )
        x_mask = mask_m[:, None] & mask_k[None, :] & in_bounds

        x_tile = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

        # ---- FMA via dot product ----
        a = x_tile.to(tl.float32)
        b = w_tile.to(tl.float32)
        acc += tl.dot(a, b, allow_tf32=True)

        k += BLOCK_K

    # Add bias if present
    bias_mask = mask_n & (has_bias > 0)
    bias_vals = tl.load(b_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias_vals[None, :]

    # Store results to output
    out_offsets = (
        n_idx[:, None] * out_n_stride
        + offs_n[None, :] * out_c_stride
        + h_out_idx[:, None] * out_h_stride
        + w_out_idx[:, None] * out_w_stride
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)


def triton_conv2d(x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor = None,
                  stride: int = 1,
                  padding: int = 0,
                  dilation: int = 1,
                  groups: int = 1) -> torch.Tensor:
    """
    High-performance 2D convolution using Triton.
    Supports arbitrary batch size, square/rectangular inputs & kernels, stride, padding, dilation.
    Grouped convs are implemented as a loop over groups in Python.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"

    # Ensure contiguous layout
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, Ci, H, W = x.shape
    Co, Ci_per_group, Kh, Kw = weight.shape
    assert Ci % groups == 0 and Ci_per_group == Ci // groups
    assert Co % groups == 0
    Co_per_group = Co // groups

    # Handle stride/padding/dilation as ints
    if isinstance(stride, tuple):
        stride_h = stride[0]
    else:
        stride_h = stride
    if isinstance(padding, tuple):
        padding_h = padding[0]
    else:
        padding_h = padding
    if isinstance(dilation, tuple):
        dilation_h = dilation[0]
    else:
        dilation_h = dilation

    # Output size (PyTorch-style conv2d formula)
    H_out = (H + 2 * padding_h - dilation_h * (Kh - 1) - 1) // stride_h + 1
    W_out = (W + 2 * padding_h - dilation_h * (Kw - 1) - 1) // stride_h + 1

    out = torch.empty((N, Co, H_out, W_out), device=x.device, dtype=x.dtype)

    # Tiling parameters
    BLOCK_M = 64  # tile in M = N*H_out*W_out
    BLOCK_N = 64  # tile in N = Co
    BLOCK_K = 32  # tile in K = Ci_per_group*Kh*Kw

    # Common scalars
    M_full = N * H_out * W_out
    has_bias = 1 if bias is not None else 0

    # Grouped convolution: run kernel per group
    for g in range(groups):
        ci_start = g * Ci_per_group
        ci_end = ci_start + Ci_per_group
        co_start = g * Co_per_group
        co_end = co_start + Co_per_group

        x_g = x[:, ci_start:ci_end, :, :]
        w_g = weight[co_start:co_end, :, :, :]
        out_g = out[:, co_start:co_end, :, :]

        if bias is not None:
            b_g = bias[co_start:co_end]
            b_ptr = b_g
        else:
            # Dummy pointer; will never be read because has_bias=0
            b_ptr = out_g

        Ci_g = Ci_per_group
        Co_g = Co_per_group
        K_g = Ci_g * Kh * Kw

        grid = lambda meta: (
            triton.cdiv(M_full, meta["BLOCK_M"]),
            triton.cdiv(Co_g, meta["BLOCK_N"]),
        )

        conv2d_fwd_kernel[grid](
            x_g,                 # x_ptr
            w_g,                 # w_ptr
            b_ptr,               # b_ptr (dummy if no bias)
            out_g,               # out_ptr
            N,                   # N
            Ci_g,                # Ci (per group)
            H,                   # H
            W,                   # W
            Co_g,                # Co (per group)
            H_out,               # H_out
            W_out,               # W_out
            Kh,                  # Kh
            Kw,                  # Kw
            stride_h,            # stride_h
            padding_h,           # padding_h
            dilation_h,          # dilation_h
            K_g,                 # K
            M_full,              # M
            has_bias,            # has_bias
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated replacement for standard 2D convolution.

    Args mirror the original Model:
        in_channels (int)
        out_channels (int)
        kernel_size (int)
        stride (int, optional)
        padding (int, optional)
        dilation (int, optional)
        groups (int, optional)
        bias (bool, optional)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = False):
        super().__init__()
        # Use nn.Conv2d only as a convenient container for weights/bias and initialization
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv2d.weight
        b = self.conv2d.bias
        stride = self.conv2d.stride[0]
        padding = self.conv2d.padding[0]
        dilation = self.conv2d.dilation[0]
        groups = self.conv2d.groups
        return triton_conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
