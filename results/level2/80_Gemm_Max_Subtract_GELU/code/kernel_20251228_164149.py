import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_gemm_bias_kernel(
    a_ptr, w_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ W + b
    A: [M, K]
    W: [K, N]   (weight transposed and made contiguous, persistent)
    b: [N]
    C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, w, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # Add bias (broadcast over rows)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(tl.float32),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def max_sub_mean_gelu_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
):
    """
    Compute:
        x_max = torch.max(x, dim=1, keepdim=True).values
        x_centered = x_max - x_max.mean(dim=1, keepdim=True)
        out = gelu(x_centered)

    For the given model configuration (max_dim == 1 and keepdim=True),
    this simplifies mathematically to out[...] = 0 for all inputs.

    We therefore just write zeros, which is both correct and maximally fast.
    """
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # We only write column 0 (keepdim=True -> shape [M, 1])
    col0 = 0
    out_ptrs = out_ptr + offs_m * stride_om + col0 * stride_on

    zeros = tl.zeros((BLOCK_M,), dtype=tl.float32)
    tl.store(out_ptrs, zeros, mask=mask_m)


def fused_linear(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    High-performance linear layer: x @ weight.T + bias

    x:        [M, K]
    weight_t: [K, N]  (persistent transposed weight, contiguous)
    bias:     [N]
    returns:  [M, N]
    """
    assert x.dim() == 2, "Input must be 2D"
    assert weight_t.dim() == 2, "weight_t must be 2D (K, N)"
    M, K = x.shape
    K_wt, N = weight_t.shape
    assert K_wt == K, "Incompatible shapes for matmul"

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_gemm_bias_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return out


def max_sub_mean_gelu_triton(x: torch.Tensor, max_dim: int) -> torch.Tensor:
    """
    Apply the sequence:
        x = torch.max(x, dim=max_dim, keepdim=True).values
        x = x - x.mean(dim=1, keepdim=True)
        x = gelu(x)

    For the target model, max_dim == 1 and keepdim=True, which implies
    the result is identically zero for any x. We implement this fast path
    via a Triton kernel that simply writes zeros with the correct shape.
    """
    assert x.dim() == 2, "Expected 2D tensor"
    assert max_dim == 1, "This optimized kernel assumes max_dim == 1"

    M, N = x.shape
    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)

    max_sub_mean_gelu_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=128,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version of the target model.

    Operations:
        1. Linear (GEMM + bias) via fused Triton kernel
        2. torch.max over dim=max_dim, keepdim=True
        3. Subtract mean over dim=1, keepdim=True
        4. GELU activation

    For the given configuration (max_dim=1), steps 2–4 simplify to an
    identically zero output; this is implemented with a specialized
    Triton kernel for maximal performance.

    Optimization:
        A persistent transposed copy of the Linear weight is maintained
        and only updated when the original weight changes, avoiding a
        costly per-forward transpose.
    """

    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim

        # Persistent transposed weight cache (not registered as parameter/buffer)
        self._weight_t = None
        self._weight_t_version = None

    def _get_weight_t(self) -> torch.Tensor:
        """
        Lazily maintain a persistent, contiguous transposed copy of self.gemm.weight.

        The cached copy is updated only when:
          * the underlying weight tensor _version changes (in-place update),
          * or device/dtype have changed.
        """
        w = self.gemm.weight
        need_update = (
            self._weight_t is None
            or self._weight_t.device != w.device
            or self._weight_t.dtype != w.dtype
            or self._weight_t_version != w._version
        )

        if need_update:
            # We do not want this transpose to be part of the autograd graph;
            # it is a pure layout optimization.
            with torch.no_grad():
                self._weight_t = w.t().contiguous()
            self._weight_t_version = w._version

        return self._weight_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to pure PyTorch on CPU (Triton is CUDA-only)
        if not x.is_cuda:
            x = self.gemm(x)
            x = torch.max(x, dim=self.max_dim, keepdim=True).values
            x = x - x.mean(dim=1, keepdim=True)
            x = torch.nn.functional.gelu(x)
            return x

        # Triton-accelerated path on CUDA
        weight_t = self._get_weight_t()
        y = fused_linear(x, weight_t, self.gemm.bias)
        out = max_sub_mean_gelu_triton(y, self.max_dim)
        return out
