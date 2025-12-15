# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---- Autotuned, high-performance fused GEMM(+bias)*scale kernel ----

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4},
            num_warps=8,
            num_stages=4,
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
    GROUP_M: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # 2D tiling with GROUP_M remapping for better L2 reuse of A / B tiles
    # -------------------------------------------------------------------------
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M

    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Guard against out-of-bounds tiles
    in_bounds_m = offs_m < M
    in_bounds_n = offs_n < N

    # -------------------------------------------------------------------------
    # Pointers for the first K tile
    # -------------------------------------------------------------------------
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Alignment hints to help Triton / compiler generate vectorized memory ops
    tl.multiple_of(offs_k, BLOCK_K)
    tl.multiple_of(a_ptrs, 16)
    tl.multiple_of(b_ptrs, 16)

    # Accumulator in fp32 (good for tf32 tensor cores as well)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # K loop: matrix multiply using tensor cores via tl.dot(allow_tf32=True)
    # -------------------------------------------------------------------------
    k_iter = 0
    while k_iter < K:
        k_remaining = K - k_iter

        a_mask = in_bounds_m[:, None] & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & in_bounds_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Uses tensor cores on Ada for fp16/bf16 or tf32 when inputs are fp32
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # -------------------------------------------------------------------------
    # Fused epilogue: (acc + bias) * scale
    #   - Only one tl.store for final result (no intermediate stores)
    # -------------------------------------------------------------------------
    # Load bias and scale per output column
    bias = tl.load(bias_ptr + offs_n, mask=in_bounds_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=in_bounds_n, other=1.0)

    # Broadcast over rows
    acc = acc + bias[None, :]
    acc = acc * scale[None, :]

    # -------------------------------------------------------------------------
    # Final store
    # -------------------------------------------------------------------------
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = in_bounds_m[:, None] & in_bounds_n[None, :]

    tl.store(c_ptrs, acc, mask=c_mask)


# ---- Python wrapper ---------------------------------------------------------


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

    # Use weight.T as B in GEMM, and make K the contiguous dimension for coalesced loads
    b = weight.t().contiguous()  # [K, N]

    # Strides
    stride_am, stride_ak = x.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = out.stride()

    # 1D grid over tiles, 2D mapping is computed inside the kernel
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']) *
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


# ---- Module wrapper ---------------------------------------------------------


class ModelNew(nn.Module):
    """
    Triton-accelerated version of the original model:

        x -> Linear -> elementwise scale -> BatchNorm1d

    GEMM + bias + scale are fused into a single Triton kernel.
    BatchNorm remains in PyTorch for correctness.
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
