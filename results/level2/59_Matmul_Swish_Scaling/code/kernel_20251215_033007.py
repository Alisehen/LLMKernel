import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Conservative baseline – good balance of occupancy and compute
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 4,
            },
            num_warps=4,
            num_stages=2,
        ),
        # Larger tile for higher compute intensity on bigger problems
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 2,
            },
            num_warps=4,
            num_stages=2,
        ),
        # More K unrolling; slightly deeper pipeline for latency hiding
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 2,
            },
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_gemm_swish_scale_kernel(
    a_ptr,        # [M, K]
    b_ptr,        # logically [K, N] (we pass weight with transposed strides)
    bias_ptr,     # [N]
    c_ptr,        # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,   # strides for logical B = weight^T
    stride_cm, stride_cn,
    scaling_factor,         # scalar float32
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # 1D launch grid with grouped ordering along M to improve L2 reuse of B
    # -------------------------------------------------------------------------
    pid = tl.program_id(axis=0)

    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % group_size
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Reusable masks for M and N dimensions
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Pointer arithmetic for the first tiles of A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Alignment / contiguity hints to Triton
    tl.multiple_of(offs_k, BLOCK_K)
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.max_contiguous(a_ptrs, BLOCK_K)
    tl.max_contiguous(b_ptrs, BLOCK_K)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # Main K loop
    # -------------------------------------------------------------------------
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # GEMM (Tensor Cores with TF32 when available)
        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # -------------------------------------------------------------------------
    # Fused epilogue: bias add + Swish + scaling
    #   Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    # -------------------------------------------------------------------------
    # Bias: [N] -> broadcast to [BLOCK_M, BLOCK_N]
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # Swish in-place with minimal temporaries
    exp_neg = tl.exp(-acc)
    acc = acc / (1.0 + exp_neg)

    # Scaling
    acc = acc * scaling_factor

    # -------------------------------------------------------------------------
    # Store result
    # -------------------------------------------------------------------------
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_linear_swish_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    """
    Fused operation:
        y = swish(x @ weight.T + bias) * scaling_factor

    Shapes:
        x:      [M, K]
        weight: [N, K]   (same layout as nn.Linear.weight)
        bias:   [N]
        return: [M, N]

    The kernel treats 'weight' as a logical [K, N] matrix (weight.T)
    by using transposed strides – no explicit transposition or copy.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA device"
    assert x.dtype == weight.dtype == bias.dtype == torch.float32, "This implementation assumes float32"

    M, K = x.shape
    N, Kw = weight.shape
    assert Kw == K, "Incompatible shapes for matmul"

    # Allocate output
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # 1D launch grid; kernel computes its own (pid_m, pid_n) with grouping.
    def grid(meta):
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        GROUP_M = meta["GROUP_M"]

        num_pid_m = triton.cdiv(M, BLOCK_M)
        num_pid_n = triton.cdiv(N, BLOCK_N)
        group_count = triton.cdiv(num_pid_m, GROUP_M)

        # Total programs = groups * GROUP_M * num_pid_n
        return (group_count * GROUP_M * num_pid_n,)

    fused_gemm_swish_scale_kernel[grid](
        x,
        weight,                 # logical B = weight^T via strides below
        bias,
        c,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(1),       # stride_bk: step along K
        weight.stride(0),       # stride_bn: step along N
        c.stride(0),
        c.stride(1),
        scaling_factor,
    )
    return c


class ModelNew(nn.Module):
    """
    High-performance replacement for a Linear + Swish + scaling block,
    backed by a single fused Triton kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.scaling_factor = float(scaling_factor)

        # Initialize similarly to nn.Linear default init
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = self.weight.size(1)
        bound = 1.0 / (fan_in ** 0.5) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_swish_scale(x, self.weight, self.bias, self.scaling_factor)
