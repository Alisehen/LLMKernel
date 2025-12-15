import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Larger tile when register pressure allows it
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        # Asymmetric tiles to trade registers for occupancy
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=4,
        ),
        # Small, very safe fallback for high register pressure
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=2,
            num_stages=3,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_scale_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N]
    bias_ptr,  # [N]
    scale_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling of the output matrix
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Common masks for row/column bounds
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Pointers for the first K tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32 for better precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply on tensor cores when possible (allow_tf32=True)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Load bias and scale (per output feature)
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=n_mask, other=1.0)

    # Fold (acc + bias) * scale into a single FMA:
    # acc = acc * scale + bias * scale
    bias = bias * scale
    acc = tl.fma(acc, scale[None, :], bias[None, :])

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_gemm_scale(x: torch.Tensor,
                     weight: torch.Tensor,
                     bias: torch.Tensor,
                     scale: torch.Tensor) -> torch.Tensor:
    """
    Fused operation:
        y = (x @ weight.T + bias) * scale

    x:      [M, K]
    weight: [N, K]  (same layout as nn.Linear.weight)
    bias:   [N]
    scale:  [N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda and scale.is_cuda, \
        "All tensors must be on CUDA for Triton kernels."

    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, "weight shape must be [N, K]"
    assert bias.shape[0] == N and scale.shape[0] == N, "bias/scale must have shape [N]"

    # Output tensor
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Use weight.T as B in GEMM, so that B is [K, N] and contiguous
    b = weight.t().contiguous()

    # Strides (assume row-major)
    stride_am, stride_ak = x.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = out.stride()

    # Kernel launch grid
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    fused_gemm_scale_kernel[grid](
        x, b, bias, scale, out,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated version:

        x -> Linear -> elementwise scale -> BatchNorm1d

    GEMM + bias + scale are fused into a single Triton kernel.
    BatchNorm is kept in PyTorch for correctness.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused GEMM + bias + scale
        y = fused_gemm_scale(x, self.gemm.weight, self.gemm.bias, self.scale)
        # BatchNorm remains in PyTorch
        y = self.bn(y)
        return y
