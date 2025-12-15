import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------
# 1. GEMM (Linear) Kernel
#    C = A @ B + bias
#    A: [M, K], B: [K, N], bias: [N], C: [M, N]
# ----------------------------

@triton.jit
def linear_gemm_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N]
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,   # a strides
    stride_bk, stride_bn,   # b strides
    stride_cm, stride_cn,   # c strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid of tiles covering C = [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers to first K-tile of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_mask = (offs_k + k) < K

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0,
        )
        # Tensor Core friendly dot (TF32 allowed for fp32 inputs)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Fused bias add (broadcast over M); same offsets/mask as C
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Store C
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask)


# ----------------------------
# 2. Row-wise Mean/Var Kernel
#    Computes mean/var over features for each row
#    Input:  x [B, N]
#    Output: mean [B], var [B]
# ----------------------------

@triton.jit
def row_mean_var_kernel(
    x_ptr,        # [B, N]
    mean_ptr,     # [B]
    var_ptr,      # [B]
    B, N,
    stride_xm, stride_xn,
    invN,         # 1.0 / N (float32)
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq_val = tl.zeros((), dtype=tl.float32)

    col = 0
    while col < N:
        cols = col + offs_n
        mask = (row < B) & (cols < N)
        x = tl.load(
            x_ptr + row * stride_xm + cols * stride_xn,
            mask=mask,
            other=0.0,
        )
        sum_val += tl.sum(x, axis=0)
        sum_sq_val += tl.sum(x * x, axis=0)
        col += BLOCK_N

    mean = sum_val * invN
    var = sum_sq_val * invN - mean * mean

    mask_row = row < B
    tl.store(mean_ptr + row, mean, mask=mask_row)
    tl.store(var_ptr + row, var, mask=mask_row)


# ----------------------------
# 3. Instance-Norm + Residual Kernel
#    y_out = ((x - mean[row]) / sqrt(var[row] + eps) + y) * y
#    Shapes: x,y,out [B, N], mean,var [B]
# ----------------------------

@triton.jit
def instancenorm_residual_kernel(
    x_ptr,        # [B, N]
    y_ptr,        # [B, N]
    mean_ptr,     # [B]
    var_ptr,      # [B]
    out_ptr,      # [B, N]
    B, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_om, stride_on,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D grid over output [B, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    mask_m = offs_m < B
    mask_n = offs_n < N

    # Load row-wise mean/var once per row tile
    mean = tl.load(mean_ptr + offs_m, mask=mask_m, other=0.0)
    var = tl.load(var_ptr + offs_m, mask=mask_m, other=0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    # Tile pointers into x, y, out
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)

    # Fused normalization + residual: all ops share same grid/offsets/mask
    x_norm = (x - mean[:, None]) * rstd[:, None]
    out = (x_norm + y) * y

    tl.store(out_ptrs, out, mask=mask)


# ----------------------------
# 4. Python Wrapper
# ----------------------------

def fused_linear_instance_norm_residual(x, y, weight, bias, eps: float):
    """
    x: [B, in_features]
    y: [B, out_features]
    weight: [out_features, in_features]
    bias: [out_features]
    returns: [B, out_features]
    """
    assert x.is_cuda and y.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == torch.float32, "This implementation assumes float32 inputs."
    assert y.dtype == torch.float32
    assert weight.dtype == torch.float32
    assert bias.dtype == torch.float32

    # Ensure contiguous for optimal memory access
    x = x.contiguous()
    y = y.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    B, K = x.shape
    out_features, K_w = weight.shape
    assert K_w == K
    assert y.shape[0] == B and y.shape[1] == out_features

    # GEMM: x @ weight.T + bias
    w_t = weight.t().contiguous()
    out_lin = torch.empty((B, out_features), device=x.device, dtype=x.dtype)

    # Tuned tile sizes for Ada (4090)
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 32

    grid_gemm = (
        triton.cdiv(B, BLOCK_M),
        triton.cdiv(out_features, BLOCK_N),
    )

    linear_gemm_kernel[grid_gemm](
        x, w_t, bias, out_lin,
        B, out_features, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        out_lin.stride(0), out_lin.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=4,
    )

    # Row-wise mean/var over features (dimension N)
    mean = torch.empty(B, device=x.device, dtype=torch.float32)
    var = torch.empty(B, device=x.device, dtype=torch.float32)
    invN = 1.0 / float(out_features)

    BLOCK_N_MV = 256  # power of 2, good balance for bandwidth/occupancy
    grid_mean_var = (B,)

    row_mean_var_kernel[grid_mean_var](
        out_lin,
        mean,
        var,
        B,
        out_features,
        out_lin.stride(0),
        out_lin.stride(1),
        invN,
        BLOCK_N=BLOCK_N_MV,
        num_warps=4,
        num_stages=2,
    )

    # Instance-norm-like normalization + residual add/mul
    out = torch.empty_like(out_lin)

    BLOCK_M_NR = 32
    BLOCK_N_NR = 256

    grid_norm = (
        triton.cdiv(B, BLOCK_M_NR),
        triton.cdiv(out_features, BLOCK_N_NR),
    )

    instancenorm_residual_kernel[grid_norm](
        out_lin,
        y,
        mean,
        var,
        out,
        B,
        out_features,
        out_lin.stride(0),
        out_lin.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
        eps,
        BLOCK_M=BLOCK_M_NR,
        BLOCK_N=BLOCK_N_NR,
        num_warps=8,
        num_stages=3,
    )

    return out


# ----------------------------
# 5. nn.Module Wrapper
# ----------------------------

class ModelNew(nn.Module):
    """
    Triton-optimized version of:
      - Linear (bmm)
      - Instance-like normalization over features
      - x = (norm(x) + y)
      - x = x * y
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.eps = eps
        self.momentum = momentum  # API parity, not used

        # Initialize like nn.Linear without extra imports
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = self.weight.size(1)
        bound = 1.0 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, y):
        return fused_linear_instance_norm_residual(x, y, self.weight, self.bias, self.eps)
