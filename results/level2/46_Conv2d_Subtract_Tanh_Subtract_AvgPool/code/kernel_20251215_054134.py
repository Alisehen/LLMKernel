import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_sub1_tanh_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, H_in, W_in,
    H_out, W_out,
    P,                      # N * H_out * W_out
    subtract1,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wk_h, stride_wk_w,
    stride_yn, stride_yc, stride_yh, stride_yw,
    C_in: tl.constexpr,     # in_channels
    C_out: tl.constexpr,    # out_channels
    K_H: tl.constexpr,      # kernel height
    K_W: tl.constexpr,      # kernel width
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Conv2d -> bias -> subtract1 -> tanh.

    Implicit-GEMM layout:
      M = N * H_out * W_out         (flattened output positions)
      N = C_out                     (output channels)
      K = C_in * K_H * K_W          (reduction dim)

    Grid:
      pid_m over M, pid_n over N.
    All fused ops (bias, subtract1, tanh, store) use the same (offs_m, offs_n, mask_m, mask_n).
    """
    pid_m = tl.program_id(0)  # over flattened output positions
    pid_n = tl.program_id(1)  # over output channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < C_out

    HW_out = H_out * W_out

    # Decode flat output index -> (n, oh, ow)
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Accumulator over [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Convolution as matmul: loop over K dimension in BLOCK_K chunks
    K_HW = K_H * K_W
    K_total = C_in * K_HW

    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_total

        # Decode K index -> (ic, kh, kw)
        ic = offs_k // K_HW
        rem_k = offs_k % K_HW
        kh = rem_k // K_W
        kw = rem_k % K_W

        # Broadcast indices
        n_b = n_idx[:, None]
        oh_b = oh_idx[:, None]
        ow_b = ow_idx[:, None]

        ic_b = ic[None, :]
        kh_b = kh[None, :]
        kw_b = kw[None, :]

        ih = oh_b + kh_b
        iw = ow_b + kw_b

        # Input tile [BLOCK_M, BLOCK_K]
        x_ptrs = (
            x_ptr
            + n_b * stride_xn
            + ic_b * stride_xc
            + ih * stride_xh
            + iw * stride_xw
        )
        # No spatial boundary checks needed for valid conv (no padding, stride=1)
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Weight tile [BLOCK_K, BLOCK_N]
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wo
            + ic[:, None] * stride_wi
            + kh[:, None] * stride_wk_h
            + kw[:, None] * stride_wk_w
        )
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # FMA over K
        acc += tl.dot(x_block, w_block)

    # Fused epilogue: bias -> subtract1 -> tanh
    # All use the SAME (offs_m, offs_n, mask_m, mask_n) grid/masks.
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b_vals[None, :]

    acc -= subtract1

    e2x = tl.exp(2.0 * acc)
    acc = (e2x - 1.0) / (e2x + 1.0)

    # Store output
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


@triton.jit
def avgpool2d_sub2_kernel(
    x_ptr, y_ptr,
    N, C, H_in, W_in,
    H_out, W_out,
    P_out,                 # N * H_out * W_out
    pool_k: tl.constexpr,
    subtract2,
    denom,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused AvgPool2d (kernel=pool_k, stride=pool_k, no padding) + subtract2.

    Grid:
      pid_m over flattened output positions (N * H_out * W_out)
      pid_n over channels

    All fused ops (subtract2, accumulation, store) use the SAME (offs_m, offs_n, mask_m, mask_n).
    """
    pid_m = tl.program_id(0)  # flattened (N, H_out, W_out)
    pid_n = tl.program_id(1)  # channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P_out
    mask_n = offs_n < C

    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    h_start = oh_idx * pool_k
    w_start = ow_idx * pool_k

    # Non-overlapping tiles, in-bounds by construction -> no spatial bounds checks
    for ph in range(0, pool_k):
        ih = h_start + ph
        ih_b = ih[:, None]

        for pw in range(0, pool_k):
            iw = w_start + pw
            iw_b = iw[:, None]

            x_ptrs = (
                x_ptr
                + n_idx[:, None] * stride_xn
                + offs_n[None, :] * stride_xc
                + ih_b * stride_xh
                + iw_b * stride_xw
            )
            x_mask = mask_m[:, None] & mask_n[None, :]

            vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
            vals = vals - subtract2
            acc += vals

    acc = acc * denom

    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


def conv2d_sub1_tanh_triton(x, weight, bias, subtract1_value):
    """
    x:       [N, C_in, H_in, W_in], float32
    weight:  [C_out, C_in, K_H, K_W], float32
    bias:    [C_out], float32
    Returns: [N, C_out, H_out, W_out]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype == torch.float32

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in_w == C_in, "Inconsistent in_channels"
    assert K_H == K_W, "Only square kernels supported"

    # Valid conv, stride=1, no padding
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1
    assert H_out > 0 and W_out > 0

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    P = N * H_out * W_out

    # Tile sizes tuned for Ada (4090) for compute-heavy conv
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    def grid(META):
        return (
            triton.cdiv(P, META["BLOCK_M"]),
            triton.cdiv(META["C_out"], META["BLOCK_N"]),
        )

    conv2d_sub1_tanh_kernel[grid](
        x, weight, bias, y,
        N, H_in, W_in,
        H_out, W_out,
        P,
        float(subtract1_value),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        C_in=C_in,
        C_out=C_out,
        K_H=K_H,
        K_W=K_W,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return y


def avgpool2d_sub2_triton(x, subtract2_value, kernel_size_pool):
    """
    x: [N, C, H_in, W_in]
    AvgPool2d with kernel_size=kernel_size_pool, stride=kernel_size_pool, padding=0.
    subtract2_value is applied before pooling and fused in the reduction.
    """
    assert x.is_cuda

    N, C, H_in, W_in = x.shape
    k = int(kernel_size_pool)

    # PyTorch-style AvgPool2d (no padding, stride=k, dilation=1)
    H_out = (H_in - k) // k + 1
    W_out = (W_in - k) // k + 1
    assert H_out > 0 and W_out > 0

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    P_out = N * H_out * W_out

    # Memory-bound kernel: prioritize large tiles and good occupancy
    BLOCK_M = 128
    BLOCK_N = 32

    def grid(META):
        return (
            triton.cdiv(P_out, META["BLOCK_M"]),
            triton.cdiv(C, META["BLOCK_N"]),
        )

    denom = float(1.0 / (k * k))

    avgpool2d_sub2_kernel[grid](
        x, y,
        N, C, H_in, W_in,
        H_out, W_out,
        P_out,
        k,
        float(subtract2_value),
        denom,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton implementation of:
        Conv2d -> subtract(subtract1_value) -> tanh -> subtract(subtract2_value) -> AvgPool2d

    Assumptions:
        - Conv2d: stride=1, padding=0, dilation=1, groups=1
        - kernel_size is an integer (square kernel)
        - AvgPool2d: kernel_size=kernel_size_pool, stride=kernel_size_pool, padding=0
        - All tensors are float32
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, tuple):
            assert kernel_size[0] == kernel_size[1], "Only square kernels supported"
            kernel_size = kernel_size[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.kernel_size_pool = int(kernel_size_pool)

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self.kernel_size, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)

    def forward(self, x):
        x = x.contiguous()
        w = self.weight.contiguous()
        b = self.bias.contiguous()

        y = conv2d_sub1_tanh_triton(x, w, b, self.subtract1_value)
        y = avgpool2d_sub2_triton(y, self.subtract2_value, self.kernel_size_pool)
        return y
