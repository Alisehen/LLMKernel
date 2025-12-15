import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_gemm_scale_kernel(
    a_ptr,        # [M, K] input
    b_ptr,        # [K, N] weight^T (contiguous)
    bias_ptr,     # [N] bias
    scale_ptr,    # [N] scale
    c_ptr,        # [M, N] output
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Multiply by scale
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)
    acc *= scale[None, :]

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_out)


def fused_gemm_scale(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K]  (nn.Linear.weight)
    bias:   [N]
    scale:  [N]
    returns [M, N] = (x @ weight.T + bias) * scale
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda and scale.is_cuda
    assert x.dtype == weight.dtype == bias.dtype == scale.dtype

    M, K = x.shape
    N = weight.shape[0]
    # weight^T: [K, N], contiguous for better access
    w_t = weight.t().contiguous()

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_gemm_scale_kernel[grid](
        x, w_t, bias, scale, c,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


class ModelNew(nn.Module):
    """
    Fused implementation of:
        y = Linear(x)
        y = y * scale
        y = BatchNorm1d(y)
    The matmul + bias + scale is done in a Triton kernel; BatchNorm1d remains in PyTorch
    to preserve training/inference behavior and running statistics.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Keep same submodule structure / parameter names as reference Model
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused gemm + bias + scale
        y = fused_gemm_scale(x, self.gemm.weight, self.gemm.bias, self.scale)
        # BatchNorm in PyTorch (training / eval semantics preserved)
        y = self.bn(y)
        return y
