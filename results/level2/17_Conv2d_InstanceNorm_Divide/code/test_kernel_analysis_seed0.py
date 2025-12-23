# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


#
# Optimized Conv2d (NCHW, stride=1, padding=0, dilation=1, groups=1)
# Implemented as implicit GEMM with autotuned tiling.
# Tuning is conservative on BLOCK_M/BLOCK_N to control register pressure.
#

@triton.autotune(
    configs=[
        # Conservative, good for register-heavy cases
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        # Larger tile when register file allows it
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=['P', 'C_out', 'Kc'],
)
@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W,
    C_out, K_h, K_w,
    H_out, W_out,
    P, Kc,  # P = N * H_out * W_out, Kc = C_in * K_h * K_w
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D tile indices
    pid_m = tl.program_id(0)  # along output positions (flattened N*H_out*W_out)
    pid_n = tl.program_id(1)  # along output channels

    # Offsets within tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Boundary masks
    m_mask = offs_m < P
    n_mask = offs_n < C_out

    # Decode flattened spatial index: P = N * H_out * W_out
    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Precompute factors for kernel indexing
    KH_KW = K_h * K_w

    # Accumulator for output tile [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Kc = C_in * K_h * K_w
    for k_start in range(0, Kc, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = offs_k < Kc

        # Map flattened K index back to (ic, kh, kw)
        ic = offs_k // KH_KW
        rem_k = offs_k % KH_KW
        kh = rem_k // K_w
        kw = rem_k % K_w

        # ------------------------------ #
        # Load input tile X  [BLOCK_M, BLOCK_K]
        # ------------------------------ #
        n_mat = n_idx[:, None]                  # [BLOCK_M, 1]
        oh_mat = oh_idx[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
        ow_mat = ow_idx[:, None] + kw[None, :]  # [BLOCK_M, BLOCK_K]
        ic_mat = ic[None, :]                    # [1, BLOCK_K]

        in_ptrs = (
            x_ptr
            + n_mat * stride_in_n
            + ic_mat * stride_in_c
            + oh_mat * stride_in_h
            + ow_mat * stride_in_w
        )

        in_mask = m_mask[:, None] & k_mask[None, :]

        x = tl.load(in_ptrs, mask=in_mask, other=0.0)

        # ------------------------------ #
        # Load weight tile W  [BLOCK_K, BLOCK_N]
        # ------------------------------ #
        ic_w = ic[:, None]          # [BLOCK_K, 1]
        kh_w = kh[:, None]          # [BLOCK_K, 1]
        kw_w = kw[:, None]          # [BLOCK_K, 1]
        oc_w = offs_n[None, :]      # [1, BLOCK_N]

        w_ptrs = (
            w_ptr
            + oc_w * stride_w_oc
            + ic_w * stride_w_ic
            + kh_w * stride_w_kh
            + kw_w * stride_w_kw
        )

        w_mask = k_mask[:, None] & n_mask[None, :]

        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # ------------------------------ #
        # Matrix multiply-accumulate
        # ------------------------------ #
        acc += tl.dot(x, w, allow_tf32=True)

    # ------------------------------ #
    # Fused bias add (broadcast over M)
    # ------------------------------ #
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)  # [BLOCK_N]
    acc += bias[None, :]  # broadcast over M

    # ------------------------------ #
    # Store output tile
    # ------------------------------ #
    out_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_out_n
        + offs_n[None, :] * stride_out_c
        + oh_idx[:, None] * stride_out_h
        + ow_idx[:, None] * stride_out_w
    )
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def conv2d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    NCHW conv2d with stride=1, padding=0, dilation=1, groups=1.
    Implemented as implicit GEMM and autotuned for RTX 4090.
    """
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is None:
        bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
    else:
        bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape

    H_out = H - K_h + 1
    W_out = W - K_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    P = N * H_out * W_out
    Kc = C_in * K_h * K_w

    def grid(meta):
        return (
            triton.cdiv(P, meta['BLOCK_M']),
            triton.cdiv(C_out, meta['BLOCK_N']),
        )

    conv2d_nchw_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W,
        C_out, K_h, K_w,
        H_out, W_out,
        P, Kc,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )

    return y


#
# Optimized InstanceNorm2d (no affine, no running stats) + division.
# Autotuned tiling with conservative BLOCK sizes to keep register pressure in check.
#

@triton.autotune(
    configs=[
        # Smaller NC vectorization, good for high register pressure
        triton.Config(
            {'BLOCK_HW': 256, 'BLOCK_NC': 4},
            num_warps=4,
            num_stages=1,
        ),
        # More NC vectorization when registers allow
        triton.Config(
            {'BLOCK_HW': 256, 'BLOCK_NC': 8},
            num_warps=4,
            num_stages=1,
        ),
        # Larger HW tile for big images
        triton.Config(
            {'BLOCK_HW': 512, 'BLOCK_NC': 4},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=['S'],  # S = H * W, reduction length
)
@triton.jit
def instance_norm_divide_kernel(
    x_ptr, y_ptr,
    N, C, H, W, S,
    eps, inv_div,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_HW: tl.constexpr,
    BLOCK_NC: tl.constexpr,  # number of (n,c) channels per program
):
    pid = tl.program_id(0)

    NC = N * C
    offs_nc = pid * BLOCK_NC + tl.arange(0, BLOCK_NC)  # [BLOCK_NC]
    nc_mask = offs_nc < NC

    n_idx = offs_nc // C
    c_idx = offs_nc % C

    # Base pointers for each (n, c)
    base_in = x_ptr + n_idx * stride_in_n + c_idx * stride_in_c
    base_out = y_ptr + n_idx * stride_out_n + c_idx * stride_out_c

    # First pass: compute mean and variance per (n, c)
    sum_ = tl.zeros((BLOCK_NC,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_NC,), dtype=tl.float32)

    for hw_start in range(0, S, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
        hw_mask = offs_hw < S

        h_idx = offs_hw // W
        w_idx = offs_hw % W

        in_ptrs = base_in[None, :] + h_idx[:, None] * stride_in_h + w_idx[:, None] * stride_in_w
        mask = hw_mask[:, None] & nc_mask[None, :]

        x = tl.load(in_ptrs, mask=mask, other=0.0)

        # Accumulate in fp32
        sum_ += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    S_f = tl.cast(S, tl.float32)
    mean = sum_ / S_f
    mean_sq = sum_sq / S_f
    var = mean_sq - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    # Combine normalization and division into a single scale factor
    scale = inv_std * inv_div

    # Second pass: normalize and divide
    for hw_start in range(0, S, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = offs_hw < S

        h_idx = offs_hw // W
        w_idx = offs_hw % W

        in_ptrs = base_in[None, :] + h_idx[:, None] * stride_in_h + w_idx[:, None] * stride_in_w
        out_ptrs = base_out[None, :] + h_idx[:, None] * stride_out_h + w_idx[:, None] * stride_out_w

        mask = hw_mask[:, None] & nc_mask[None, :]

        x = tl.load(in_ptrs, mask=mask, other=0.0)
        # y = (x - mean) * scale
        y = (x - mean[None, :]) * scale[None, :]
        tl.store(out_ptrs, y, mask=mask)


def instance_norm_divide_triton(x: torch.Tensor, eps: float, divide_by: float) -> torch.Tensor:
    """
    Instance normalization over HxW for each (N, C), followed by division by a scalar.
    Equivalent to nn.InstanceNorm2d (affine=False, track_running_stats=False) and x / divide_by.
    """
    x = x.contiguous()
    N, C, H, W = x.shape
    y = torch.empty_like(x)

    S = H * W
    inv_div = 1.0 / float(divide_by)

    def grid(meta):
        NC = N * C
        return (triton.cdiv(NC, meta['BLOCK_NC']),)

    instance_norm_divide_kernel[grid](
        x, y,
        N, C, H, W, S,
        eps, inv_div,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of:
      Conv2d -> InstanceNorm2d (no affine) -> division by constant
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Use PyTorch instance norm only for eps and state_dict compatibility.
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = float(divide_by)

    def forward(self, x):
        # x expected to be on CUDA for Triton kernels
        y = conv2d_triton(x, self.conv.weight, self.conv.bias)
        y = instance_norm_divide_triton(y, self.instance_norm.eps, self.divide_by)
        return y
