# complete ModelNew code with optimized Triton kernels

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------
# 1) GEMM (Linear) kernel: y = x @ W^T + b
# ---------------------------
@triton.jit
def linear_gemm_kernel(
    a_ptr, w_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < (K - k)
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, w, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # add bias: shape [N], broadcast over rows
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_out)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K]  (nn.Linear weight)
    bias:   [N]
    returns y = x @ weight.T + bias  with shape [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    # W^T is [K, N], contiguous for better memory coalescing
    w_t = weight.t().contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            max(1, triton.cdiv(M, meta["BLOCK_M"])),
            max(1, triton.cdiv(N, meta["BLOCK_N"])),
        )

    linear_gemm_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(1), w_t.stride(0),
        y.stride(0), y.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        num_warps=8, num_stages=2,
    )
    return y


# ---------------------------
# 2) Max-reduction kernels
# ---------------------------
@triton.jit
def rowwise_max_kernel(
    in_ptr, out_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    # Each program processes one row (reduce over columns)
    pid = tl.program_id(0)
    row = pid
    row_mask = row < M

    row_in_ptr = in_ptr + row * stride_im
    row_out_ptr = out_ptr + row * stride_om

    # scalar accumulator in fp32
    max_val = tl.full((), -float("inf"), dtype=tl.float32)

    cols = tl.arange(0, BLOCK_N)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + cols
        ptrs = row_in_ptr + offs_n * stride_in
        vals = tl.load(
            ptrs,
            mask=row_mask & (offs_n < N),
            other=-float("inf"),
        )
        vals_f32 = vals.to(tl.float32)
        block_max = tl.max(vals_f32, axis=0)  # scalar
        max_val = tl.maximum(max_val, block_max)

    # store scalar result
    tl.store(row_out_ptr, max_val, mask=row_mask)


@triton.jit
def colwise_max_kernel(
    in_ptr, out_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
):
    # Each program processes one column (reduce over rows)
    pid = tl.program_id(0)
    col = pid
    col_mask = col < N

    col_out_ptr = out_ptr + col * stride_on

    max_val = tl.full((), -float("inf"), dtype=tl.float32)

    rows = tl.arange(0, BLOCK_M)
    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + rows
        ptrs = in_ptr + offs_m * stride_im + col * stride_in
        vals = tl.load(
            ptrs,
            mask=col_mask & (offs_m < M),
            other=-float("inf"),
        )
        vals_f32 = vals.to(tl.float32)
        block_max = tl.max(vals_f32, axis=0)  # scalar
        max_val = tl.maximum(max_val, block_max)

    tl.store(col_out_ptr, max_val, mask=col_mask)


def triton_max_reduce(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: [M, N]
    dim: 0 or 1
    returns torch.max(x, dim=dim, keepdim=True).values
    """
    assert x.is_cuda
    M, N = x.shape
    if dim == 1:
        # Reduce over columns: output [M, 1]
        out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

        def grid(meta):
            return (max(1, triton.cdiv(M, 1)),)

        rowwise_max_kernel[grid](
            x, out,
            M, N,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_N=128,
            num_warps=4, num_stages=2,
        )
        return out
    elif dim == 0:
        # Reduce over rows: output [1, N]
        out = torch.empty((1, N), device=x.device, dtype=x.dtype)

        def grid(meta):
            return (max(1, triton.cdiv(N, 1)),)

        colwise_max_kernel[grid](
            x, out,
            M, N,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=128,
            num_warps=4, num_stages=2,
        )
        return out
    else:
        raise NotImplementedError("Only dim=0 or dim=1 are supported for max reduction.")


# ---------------------------
# 3) Mean-subtract + GELU kernel
#    out = GELU(z - z.mean(dim=1, keepdim=True))
# ---------------------------
@triton.jit
def mean_sub_gelu_kernel(
    in_ptr, out_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    row_mask = row < M

    row_in_ptr = in_ptr + row * stride_im
    row_out_ptr = out_ptr + row * stride_om

    # ---- First pass: compute row-wise mean over dim=1 ----
    sum_val = tl.zeros((), dtype=tl.float32)

    cols = tl.arange(0, BLOCK_N)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + cols
        ptrs = row_in_ptr + offs_n * stride_in
        vals = tl.load(
            ptrs,
            mask=row_mask & (offs_n < N),
            other=0.0,
        )
        vals_f32 = vals.to(tl.float32)
        block_sum = tl.sum(vals_f32, axis=0)  # scalar
        sum_val += block_sum

    n_f32 = tl.full((), N, dtype=tl.float32)
    mean = sum_val / n_f32  # scalar

    # ---- Second pass: subtract mean and apply GELU ----
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + cols
        ptrs = row_in_ptr + offs_n * stride_in
        vals = tl.load(
            ptrs,
            mask=row_mask & (offs_n < N),
            other=0.0,
        )
        x = vals.to(tl.float32) - mean  # center

        # GELU approximation:
        # gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
        x3 = x * x * x
        u = 0.7978845608028654 * (x + 0.044715 * x3)  # sqrt(2/pi) ≈ 0.79788456
        e = tl.exp(2.0 * u)
        tanh_u = (e - 1.0) / (e + 1.0)
        gelu = 0.5 * x * (1.0 + tanh_u)

        out_vals = gelu.to(vals.dtype)
        out_ptrs = row_out_ptr + offs_n * stride_on
        tl.store(out_ptrs, out_vals, mask=row_mask & (offs_n < N))


def triton_mean_sub_gelu(z: torch.Tensor) -> torch.Tensor:
    """
    z: 2D tensor
       In this model, z is either [B, 1] (if max_dim=1) or [1, F] (if max_dim=0).
    returns GELU(z - z.mean(dim=1, keepdim=True))
    """
    assert z.is_cuda
    M, N = z.shape
    out = torch.empty_like(z)

    def grid(meta):
        return (max(1, triton.cdiv(M, 1)),)

    mean_sub_gelu_kernel[grid](
        z, out,
        M, N,
        z.stride(0), z.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=128,
        num_warps=4, num_stages=2,
    )
    return out


# ---------------------------
# 4) High-performance replacement model
# ---------------------------
class ModelNew(nn.Module):
    """
    Model that performs:
      1) GEMM (Linear)
      2) torch.max(x, dim=max_dim, keepdim=True).values
      3) subtraction of mean along dim=1 (keepdim=True)
      4) GELU activation

    All heavy ops are implemented with Triton kernels.
    """

    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.max_dim = max_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move parameters to same device as input if needed
        if self.weight.device != x.device:
            self.weight.data = self.weight.data.to(x.device)
            self.bias.data = self.bias.data.to(x.device)

        # 1) GEMM
        x = triton_linear(x, self.weight, self.bias)

        # 2) Max reduction with keepdim
        x = triton_max_reduce(x, self.max_dim)

        # 3) Subtract mean over dim=1 and 4) GELU
        x = triton_mean_sub_gelu(x)
        return x
