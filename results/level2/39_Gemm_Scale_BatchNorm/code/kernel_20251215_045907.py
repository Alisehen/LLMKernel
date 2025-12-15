# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Balanced tiles
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=4,
        ),
        # Larger tiles for big matrices – better TC utilization
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=16,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=16,
            num_stages=4,
        ),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Program id and swizzled 2D tile coordinates (pid_m, pid_n)
    # Swizzling along M improves L2 cache reuse of B across tiles.
    # -------------------------------------------------------------------------
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    # -------------------------------------------------------------------------
    # Offsets for this tile
    # All fused ops (bias + scale + store) share offs_m/offs_n and mask_out.
    # -------------------------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Hints to compiler for alignment / vectorization
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    # Base pointers for A and B tiles
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator in FP32 for better precision, will downcast on store if needed
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Static masks on M and N (shared by all ops in the tile)
    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    # -------------------------------------------------------------------------
    # Main K loop
    # -------------------------------------------------------------------------
    k = 0
    while k < K:
        k_offsets = k + offs_k

        # Load A and B with per-iteration K boundary
        a = tl.load(
            a_ptrs,
            mask=mask_m & (k_offsets[None, :] < K),
            other=0.0,
            cache_modifier=".cg",  # stream A through L2
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < K) & mask_n,
            other=0.0,
            cache_modifier=".ca",  # keep B hot in cache (reused across M tiles)
        )

        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # -------------------------------------------------------------------------
    # Fused epilogue: bias add + scale
    # All elementwise ops use offs_n and masks derived from mask_n.
    # -------------------------------------------------------------------------
    # Load bias and scale once per output column tile
    bias = tl.load(
        bias_ptr + offs_n,
        mask=offs_n < N,
        other=0.0,
        cache_modifier=".ca",
    )
    scale = tl.load(
        scale_ptr + offs_n,
        mask=offs_n < N,
        other=1.0,
        cache_modifier=".ca",
    )

    # Broadcast over rows of the tile
    acc = acc + bias[None, :]
    acc = acc * scale[None, :]

    # -------------------------------------------------------------------------
    # Store result
    # -------------------------------------------------------------------------
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_out = mask_m & mask_n
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

    # weight^T: [K, N], contiguous for best access from Triton
    w_t = weight.t().contiguous()

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        # 1D grid over (M_tiles * N_tiles), with swizzling handled in kernel
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

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

    The matmul + bias + scale are executed in a single optimized Triton kernel.
    BatchNorm1d remains in PyTorch to preserve training/inference semantics
    and running statistics behavior.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused GEMM + bias + scale in Triton
        y = fused_gemm_scale(x, self.gemm.weight, self.gemm.bias, self.scale)
        # BatchNorm in PyTorch
        y = self.bn(y)
        return y
