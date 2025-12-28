# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ============================================
# Fused 3D Convolution + Min over Depth (dim=2)
# Autotuned BLOCK sizes with register-pressure awareness
# ============================================
@triton.autotune(
    configs=[
        # Larger, more compute-dense tile – good when registers allow
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_KCI": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        # More conservative on registers – higher occupancy fallback
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_KCI": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        # Very conservative fallback for extreme register pressure
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 16,
                "BLOCK_KCI": 32,
            },
            num_warps=4,
            num_stages=1,
        ),
    ],
    key=["Cin", "Co", "D_out", "H_out", "W_out"],
)
@triton.jit
def conv3d_min_depth_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, Co,
    D_in, H_in, W_in,
    kD, kH, kW,
    D_out, H_out, W_out,
    stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kd, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    M_no_depth,                     # = N * H_out * W_out
    BLOCK_M: tl.constexpr,          # tile over (N * H_out * W_out)
    BLOCK_N: tl.constexpr,          # tile over output channels Co
    BLOCK_KCI: tl.constexpr,        # tile over Cin
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets in the flattened (N * H_out * W_out) and Co dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M_no_depth
    mask_n = offs_n < Co

    # Decode flattened spatial index offs_m -> (n, oh, ow)
    W_out_ = W_out
    H_out_ = H_out

    ow = offs_m % W_out_
    tmp = offs_m // W_out_
    oh = tmp % H_out_
    n = tmp // H_out_

    # Preload bias for this BLOCK_N of output channels
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)

    # Running minimum over depth dimension (od)
    min_acc = tl.full((BLOCK_M, BLOCK_N), float("inf"), dtype=tl.float32)

    od = 0
    while od < D_out:
        # Base input pointer for each (n, od, oh, ow) at channel=0
        base_in = (
            x_ptr
            + n * stride_x_n
            + od * stride_x_d
            + oh * stride_x_h
            + ow * stride_x_w
        )

        # Accumulator for this depth slice
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        kd = 0
        while kd < kD:
            base_in_kd = base_in + kd * stride_x_d
            kh = 0
            while kh < kH:
                base_in_kh = base_in_kd + kh * stride_x_h
                kw = 0
                while kw < kW:
                    base_in_kw = base_in_kh + kw * stride_x_w

                    ci_start = 0
                    while ci_start < Cin:
                        offs_ci = ci_start + tl.arange(0, BLOCK_KCI)
                        mask_ci = offs_ci < Cin

                        # Load input tile: shape (BLOCK_M, BLOCK_KCI)
                        a_ptrs = base_in_kw[:, None] + offs_ci[None, :] * stride_x_c
                        a = tl.load(
                            a_ptrs,
                            mask=mask_m[:, None] & mask_ci[None, :],
                            other=0.0,
                        )

                        # Load weight tile: shape (BLOCK_KCI, BLOCK_N)
                        # w layout: (Co, Cin, kD, kH, kW)
                        b_ptrs = (
                            w_ptr
                            + offs_n[None, :] * stride_w_co
                            + offs_ci[:, None] * stride_w_ci
                            + kd * stride_w_kd
                            + kh * stride_w_kh
                            + kw * stride_w_kw
                        )
                        b = tl.load(
                            b_ptrs,
                            mask=mask_ci[:, None] & mask_n[None, :],
                            other=0.0,
                        )

                        # Matmul for this (kd, kh, kw, ci block)
                        acc += tl.dot(a, b, allow_tf32=True)

                        ci_start += BLOCK_KCI
                    kw += 1
                kh += 1
            kd += 1

        # Add bias and update running minimum over depth
        acc += bias[None, :]
        min_acc = tl.minimum(min_acc, acc)

        od += 1

    # Store final min over depth to y[n, co, oh, ow]
    y_ptrs = (
        y_ptr
        + n[:, None] * stride_y_n
        + offs_n[None, :] * stride_y_c
        + oh[:, None] * stride_y_h
        + ow[:, None] * stride_y_w
    )
    tl.store(
        y_ptrs,
        min_acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def conv3d_min_depth_triton(x, weight, bias, kernel_size, dim):
    """
    Fused 3D convolution + min over depth (dim=2).

    Args:
        x:       (N, Cin, D_in, H_in, W_in)
        weight:  (Co, Cin, kD, kH, kW)
        bias:    (Co,)
        kernel_size: int or (kD, kH, kW)
        dim: must be 2 (depth)

    Returns:
        y: (N, Co, H_out, W_out) with min over depth applied.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dim() == 5 and weight.dim() == 5
    assert dim == 2, "Fused kernel reduces along depth dimension (dim=2) only."

    if isinstance(kernel_size, int):
        kD = kH = kW = kernel_size
    else:
        kD, kH, kW = kernel_size

    N, Cin, D_in, H_in, W_in = x.shape
    Co = weight.shape[0]

    # 'valid' convolution output sizes
    D_out = D_in - kD + 1
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1

    # Output tensor: min over depth already applied
    y = torch.empty((N, Co, H_out, W_out), device=x.device, dtype=x.dtype)

    M_no_depth = N * H_out * W_out

    grid = lambda META: (
        triton.cdiv(M_no_depth, META["BLOCK_M"]),
        triton.cdiv(Co, META["BLOCK_N"]),
    )

    conv3d_min_depth_kernel[grid](
        x, weight, bias, y,
        N, Cin, Co,
        D_in, H_in, W_in,
        kD, kH, kW,
        D_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2),
        weight.stride(3), weight.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        M_no_depth,
    )
    return y


# ============================================
# Softmax along channel dimension (dim=1) for (N, C, H, W)
# Autotuned BLOCK_C for 4090
# ============================================
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_C": 256,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_C": 128,
            },
            num_warps=4,
            num_stages=2,
        ),
        # Smaller fallback when register pressure or very small C
        triton.Config(
            {
                "BLOCK_C": 64,
            },
            num_warps=2,
            num_stages=1,
        ),
    ],
    key=["C"],
)
@triton.jit
def softmax_dim1_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    BLOCK_C: tl.constexpr,
):
    # Each program computes softmax over channels for one (n, h, w)
    pid = tl.program_id(0)
    n_rows = N * H * W
    row = pid
    mask_row = row < n_rows

    # Decode row -> (n, h, w)
    w_idx = row % W
    tmp = row // W
    h_idx = tmp % H
    n_idx = tmp // H

    base_x = (
        x_ptr
        + n_idx * stride_x_n
        + h_idx * stride_x_h
        + w_idx * stride_x_w
    )
    base_y = (
        y_ptr
        + n_idx * stride_y_n
        + h_idx * stride_y_h
        + w_idx * stride_y_w
    )

    offs_c = tl.arange(0, BLOCK_C)

    # 1st pass: compute max over channels
    row_max = tl.full((1,), -float("inf"), dtype=tl.float32)
    c_start = 0
    while c_start < C:
        offs = c_start + offs_c
        mask_c = offs < C
        mask = mask_row & mask_c
        x_ptrs = base_x + offs * stride_x_c
        x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        chunk_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, chunk_max)
        c_start += BLOCK_C

    # 2nd pass: compute denominator (sum of exp)
    row_sum = tl.zeros((1,), dtype=tl.float32)
    c_start = 0
    while c_start < C:
        offs = c_start + offs_c
        mask_c = offs < C
        mask = mask_row & mask_c
        x_ptrs = base_x + offs * stride_x_c
        x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        x = x - row_max
        num = tl.exp(x)
        chunk_sum = tl.sum(num, axis=0)
        row_sum += chunk_sum
        c_start += BLOCK_C

    # 3rd pass: write normalized probabilities
    c_start = 0
    while c_start < C:
        offs = c_start + offs_c
        mask_c = offs < C
        mask = mask_row & mask_c
        x_ptrs = base_x + offs * stride_x_c
        x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        x = x - row_max
        num = tl.exp(x)
        out = num / row_sum
        y_ptrs = base_y + offs * stride_y_c
        tl.store(y_ptrs, out, mask=mask)
        c_start += BLOCK_C


def softmax_dim1_triton(x):
    """
    x: (N, C, H, W)
    Softmax along channel dimension C (dim=1).
    """
    assert x.is_cuda
    assert x.dim() == 4

    N, C, H, W = x.shape
    y = torch.empty_like(x)

    n_rows = N * H * W
    grid = lambda META: (max(1, n_rows),)

    softmax_dim1_kernel[grid](
        x, y,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )
    return y


# ============================================
# ModelNew: Triton-accelerated replacement
# ============================================
class ModelNew(nn.Module):
    """
    Triton implementation of:
      - 3D convolution
      - min over depth dimension (dim=2)
      - softmax over channel dimension (dim=1)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        # Support int or tuple kernel_size, like nn.Conv3d
        if isinstance(kernel_size, int):
            kD = kH = kW = kernel_size
        else:
            kD, kH, kW = kernel_size
        self.kernel_size = (kD, kH, kW)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim  # expected to be 2 (depth)

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kD, kH, kW)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # Fused 3D convolution + min over depth (dim=2)
        x = conv3d_min_depth_triton(x, self.weight, self.bias, self.kernel_size, self.dim)
        # Softmax along channel dimension (dim=1)
        x = softmax_dim1_triton(x)
        return x
