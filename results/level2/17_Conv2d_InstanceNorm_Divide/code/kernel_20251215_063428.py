import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------------
# Optimized Conv2D NCHW Kernel
# -------------------------------

@triton.autotune(
    configs=[
        # Small, low-register tile: best for high occupancy / large C_in*KH*KW
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32},
            num_warps=2,
            num_stages=3,
        ),
        # Asymmetric tiles to better balance OC vs spatial when shapes differ
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["N", "OC", "H_out", "W_out"],
)
@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W,
    OC, KH, KW,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    # meta-parameters
    BLOCK_M: tl.constexpr,  # flattened output positions (N * H_out * W_out)
    BLOCK_N: tl.constexpr,  # output channels
):
    # 2D grid over (output positions, output channels)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    P = N * H_out * W_out

    mask_m = offs_m < P
    mask_n = offs_n < OC

    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # Base pointers for this tile
    x_base = (
        x_ptr
        + n_idx * stride_xn
        + oh_idx * stride_xh
        + ow_idx * stride_xw
    )  # [BM]
    w_base = w_ptr + offs_n * stride_wo  # [BN]

    # Accumulator tile kept in registers; BLOCK_* kept moderate to avoid spills
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Convolution as implicit GEMM: sum over K = C_in * KH * KW
    # Keep inner body minimal to limit register lifetimes
    for ic in range(0, C_in):
        x_ic_base = x_base + ic * stride_xc
        w_ic_base = w_base + ic * stride_wc

        for kh in range(0, KH):
            x_ich_base = x_ic_base + kh * stride_xh
            w_ich_base = w_ic_base + kh * stride_wkh

            for kw in range(0, KW):
                x_ptrs = x_ich_base + kw * stride_xw      # [BM]
                w_ptrs = w_ich_base + kw * stride_wkw     # [BN]

                x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0)          # [BM]
                w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0)          # [BN]

                # Accumulate outer product
                acc += x_vals[:, None] * w_vals[None, :]

    # Bias add
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)  # [BN]
    acc += bias[None, :]

    # Store output
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask)


# -------------------------------
# InstanceNorm2D: Stats Kernel
# -------------------------------

@triton.jit
def instance_norm2d_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    N, C, H, W,
    eps,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_meann, stride_meanc,
    stride_rstdn, stride_rstdc,
    BLOCK_HW: tl.constexpr,
):
    # One program per (n, c) computes stats over H*W
    pid = tl.program_id(0)
    NC = N * C
    if pid >= NC:
        return

    c = pid % C
    n = pid // C

    P = H * W

    sum_val = tl.zeros((), dtype=tl.float32)
    sq_sum_val = tl.zeros((), dtype=tl.float32)

    offs_base = 0
    # Stream through H*W; memory-bound, so keep inner loop simple
    while offs_base < P:
        offs_hw = offs_base + tl.arange(0, BLOCK_HW)
        mask = offs_hw < P

        oh = offs_hw // W
        ow = offs_hw % W

        x_ptrs = (
            x_ptr
            + n * stride_xn
            + c * stride_xc
            + oh * stride_xh
            + ow * stride_xw
        )
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        sum_val += tl.sum(x_f32, axis=0)
        sq_sum_val += tl.sum(x_f32 * x_f32, axis=0)

        offs_base += BLOCK_HW

    P_f32 = tl.full((), P, dtype=tl.float32)
    mean = sum_val / P_f32
    var = sq_sum_val / P_f32 - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    mean_ptr_nc = mean_ptr + n * stride_meann + c * stride_meanc
    rstd_ptr_nc = rstd_ptr + n * stride_rstdn + c * stride_rstdc
    tl.store(mean_ptr_nc, mean)
    tl.store(rstd_ptr_nc, rstd)


# -------------------------------
# InstanceNorm2D: Apply + Divide Kernel
# -------------------------------

@triton.autotune(
    configs=[
        # Balanced low-register tile
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32},
            num_warps=2,
            num_stages=2,
        ),
        # Larger tile along M (N*H*W) when C is small/moderate
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Larger tile along N (channels) when C is large
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["N", "C", "H", "W"],
)
@triton.jit
def instance_norm2d_apply_div_kernel(
    x_ptr, mean_ptr, rstd_ptr, y_ptr,
    N, C, H, W,
    divide_by,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_meann, stride_meanc,
    stride_rstdn, stride_rstdc,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # flattened positions N*H*W
    BLOCK_N: tl.constexpr,  # channels
):
    # 2D grid over (flattened spatial+batch, channels)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    P = N * H * W

    mask_m = offs_m < P
    mask_n = offs_n < C

    HW = H * W
    n_idx = offs_m // HW
    rem = offs_m % HW
    h_idx = rem // W
    w_idx = rem % W

    x_base = (
        x_ptr
        + n_idx[:, None] * stride_xn
        + h_idx[:, None] * stride_xh
        + w_idx[:, None] * stride_xw
    )
    y_base = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + h_idx[:, None] * stride_yh
        + w_idx[:, None] * stride_yw
    )
    mean_base = mean_ptr + n_idx[:, None] * stride_meann
    rstd_base = rstd_ptr + n_idx[:, None] * stride_rstdn

    x_ptrs = x_base + offs_n[None, :] * stride_xc
    y_ptrs = y_base + offs_n[None, :] * stride_yc
    mean_ptrs = mean_base + offs_n[None, :] * stride_meanc
    rstd_ptrs = rstd_base + offs_n[None, :] * stride_rstdc

    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    mean = tl.load(mean_ptrs, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptrs, mask=mask, other=0.0)

    # Cheap ops; keep expression compact to reduce live ranges
    y = (x_f32 - mean) * rstd
    y = y / divide_by

    tl.store(y_ptrs, y, mask=mask)


# -------------------------------
# Python Wrappers
# -------------------------------

def triton_conv2d_nchw(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype == torch.float32
    assert x.ndim == 4  # N, C_in, H, W
    assert weight.ndim == 4  # OC, C_in, KH, KW

    N, C_in, H, W = x.shape
    OC, Cw, KH, KW = weight.shape
    assert Cw == C_in

    # Only support stride=1, padding=0, dilation=1
    H_out = H - KH + 1
    W_out = W - KW + 1
    assert H_out > 0 and W_out > 0

    y = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    stride_xn, stride_xc, stride_xh, stride_xw = x_contig.stride()
    stride_wo, stride_wc, stride_wkh, stride_wkw = w_contig.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    P = N * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(OC, meta["BLOCK_N"]),
        )

    conv2d_nchw_kernel[grid](
        x_contig, w_contig, b_contig, y,
        N, C_in, H, W,
        OC, KH, KW,
        H_out, W_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wo, stride_wc, stride_wkh, stride_wkw,
        stride_yn, stride_yc, stride_yh, stride_yw,
    )

    return y


def triton_instance_norm2d_divide(x: torch.Tensor, eps: float, divide_by: float) -> torch.Tensor:
    assert x.is_cuda
    assert x.dtype == torch.float32
    assert x.ndim == 4  # N, C, H, W

    N, C, H, W = x.shape
    x_contig = x.contiguous()

    mean = torch.empty((N, C), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N, C), device=x.device, dtype=torch.float32)

    stride_xn, stride_xc, stride_xh, stride_xw = x_contig.stride()
    stride_meann, stride_meanc = mean.stride()
    stride_rstdn, stride_rstdc = rstd.stride()

    # Memory-bound; choose reasonably large tile for HW
    BLOCK_HW = 1024
    grid_stats = (N * C,)

    instance_norm2d_stats_kernel[grid_stats](
        x_contig, mean, rstd,
        N, C, H, W,
        eps,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_meann, stride_meanc,
        stride_rstdn, stride_rstdc,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=1,
    )

    y = torch.empty_like(x_contig)

    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    P = N * H * W

    def grid_apply(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C, meta["BLOCK_N"]),
        )

    instance_norm2d_apply_div_kernel[grid_apply](
        x_contig, mean, rstd, y,
        N, C, H, W,
        divide_by,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_meann, stride_meanc,
        stride_rstdn, stride_rstdc,
        stride_yn, stride_yc, stride_yh, stride_yw,
    )

    return y


# -------------------------------
# Model
# -------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.eps = 1e-5
        self.divide_by = float(divide_by)

    def forward(self, x):
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        weight = self.conv.weight
        bias = self.conv.bias
        x = triton_conv2d_nchw(x, weight, bias)
        x = triton_instance_norm2d_divide(x, self.eps, self.divide_by)
        return x
