import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------- GEMM: Linear Projection ------------------------- #

@triton.autotune(
    configs=[
        # Conservative baseline: good occupancy, low register pressure
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Higher throughput for more square-ish shapes when registers allow
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        # Skewed toward tall matrices (M >> N)
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N]
    c_ptr,  # [M, N]
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Precompute masks on invariant dimensions
    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        a = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def triton_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x: [M, K]
    weight: [K, N]
    returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == weight.dtype == torch.float32, "Only float32 supported for now"

    x = x.contiguous()
    w = weight.contiguous()

    M, K = x.shape
    K_w, N = w.shape
    assert K_w == K, "Incompatible dimensions for matmul"

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    _linear_kernel[grid](
        x,
        w,
        c,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        c.stride(0),
        c.stride(1),
    )

    return c


# ------------------------ Fused VLAD Aggregation --------------------------- #

@triton.autotune(
    configs=[
        # Baseline: balanced tile sizes, low register pressure
        triton.Config(
            {"BLOCK_D": 64, "BLOCK_K": 64, "BLOCK_N": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Larger D tile for feature-rich descriptors, still conservative staging
        triton.Config(
            {"BLOCK_D": 128, "BLOCK_K": 64, "BLOCK_N": 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["D", "K", "N"],
)
@triton.jit
def _vlad_fused_kernel(
    x_ptr,          # [B, N, D]
    assign_ptr,     # [B, N, K]
    clusters2_ptr,  # [D, K]
    vlad_ptr,       # [B, D, K]
    B,
    N,
    D,
    K,
    stride_xb,
    stride_xn,
    stride_xd,
    stride_ab,
    stride_an,
    stride_ak,
    stride_c2d,
    stride_c2k,
    stride_vb,
    stride_vd,
    stride_vk,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused VLAD kernel:

      v[b, d, k] = sum_n assign[b, n, k] * (x[b, n, d] - clusters2[d, k])

    Computes, per (b, d, k) tile:
      - GEMM-like accumulation over N
      - Sum over assignments for each cluster (for centering term)
      - Subtraction of cluster centers
    """
    pid_d = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    # Invariant masks for D and K
    d_mask = offs_d < D
    k_mask = offs_k < K

    x_batch_ptr = x_ptr + pid_b * stride_xb
    a_batch_ptr = assign_ptr + pid_b * stride_ab
    v_batch_ptr = vlad_ptr + pid_b * stride_vb

    # (D x N) @ (N x K) -> (D x K) with extra reduction on assignments
    x_ptrs = x_batch_ptr + offs_d[:, None] * stride_xd + offs_n[None, :] * stride_xn
    asg_ptrs = a_batch_ptr + offs_n[:, None] * stride_an + offs_k[None, :] * stride_ak

    acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
    acc_sum = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_remaining = N - n
        n_mask = offs_n < n_remaining

        x_tile = tl.load(
            x_ptrs,
            mask=d_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        asg_tile = tl.load(
            asg_ptrs,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        acc += tl.dot(x_tile, asg_tile, allow_tf32=True)
        acc_sum += tl.sum(asg_tile, axis=0)

        x_ptrs += BLOCK_N * stride_xn
        asg_ptrs += BLOCK_N * stride_an

    # Load cluster centers: [D, K] tile
    c2_ptrs = clusters2_ptr + offs_d[:, None] * stride_c2d + offs_k[None, :] * stride_c2k
    valid = d_mask[:, None] & k_mask[None, :]
    c2 = tl.load(c2_ptrs, mask=valid, other=0.0)

    # Broadcast acc_sum over D and subtract centering term
    s = acc_sum[None, :]  # [1, BLOCK_K]
    v_tile = acc - c2 * s

    v_ptrs = v_batch_ptr + offs_d[:, None] * stride_vd + offs_k[None, :] * stride_vk
    tl.store(v_ptrs, v_tile, mask=valid)


def triton_vlad_fused(
    x: torch.Tensor,          # [B, N, D]
    assignment: torch.Tensor, # [B, N, K]
    clusters2: torch.Tensor,  # [1, D, K]
) -> torch.Tensor:
    """
    Compute VLAD descriptor:

      v[b, d, k] = sum_n assignment[b, n, k] * (x[b, n, d] - clusters2[d, k])

    using a single high-performance Triton kernel.
    Returns: [B, D, K]
    """
    assert x.is_cuda and assignment.is_cuda and clusters2.is_cuda
    assert x.dtype == assignment.dtype == clusters2.dtype == torch.float32

    x = x.contiguous()
    assignment = assignment.contiguous()

    B, N, D = x.shape
    B_a, N_a, K = assignment.shape
    assert B_a == B and N_a == N, "Incompatible shapes for x and assignment"

    # clusters2: [1, D, K] -> [D, K]
    clusters2_2d = clusters2.view(D, K)

    vlad = torch.empty((B, D, K), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(D, meta["BLOCK_D"]),
            triton.cdiv(K, meta["BLOCK_K"]),
            B,
        )

    _vlad_fused_kernel[grid](
        x,
        assignment,
        clusters2_2d,
        vlad,
        B,
        N,
        D,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        assignment.stride(0),
        assignment.stride(1),
        assignment.stride(2),
        clusters2_2d.stride(0),
        clusters2_2d.stride(1),
        vlad.stride(0),
        vlad.stride(1),
        vlad.stride(2),
    )

    return vlad


# ----------------------------- Model Definition ---------------------------- #

class ModelNew(nn.Module):
    """
    Triton-accelerated NetVLAD-like module.

    Optimizations:
      - First projection (BN*D -> BN*(K+G)) via high-performance Triton GEMM.
      - VLAD aggregation fused into a single Triton kernel:
          v[b, d, k] = sum_n a[b, n, k] * (x[b, n, d] - c[d, k])
        eliminating separate sum, broadcast, and batched matmul.
    """

    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = 1.0 / (feature_size ** 0.5)
        clusters = cluster_size + ghost_clusters

        # Projection weights (w, b) in the paper
        self.clusters = nn.Parameter(
            init_sc * torch.randn(feature_size, clusters)
        )
        self.batch_norm = nn.BatchNorm1d(clusters)

        # Visual words c_k: [1, D, K]
        self.clusters2 = nn.Parameter(
            init_sc * torch.randn(1, feature_size, cluster_size)
        )

        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """
        x: B x N x D
        returns: B x (D*K)
        """
        B, max_sample, D = x.shape
        assert D == self.feature_size

        # Flatten over batch and time: [B*N, D]
        x_flat = x.view(-1, self.feature_size).contiguous()

        if x_flat.device != self.clusters.device:
            msg = f"x.device {x_flat.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # BN x (K+G) using Triton GEMM
        assignment = triton_linear(x_flat, self.clusters)

        # BatchNorm over feature dimension (clusters)
        assignment = self.batch_norm(assignment)

        # Softmax over clusters (K+G)
        assignment = torch.nn.functional.softmax(assignment, dim=1)

        # Remove ghost clusters: BN x K
        assignment = assignment[:, : self.cluster_size]

        # BN x K -> B x N x K
        assignment = assignment.view(B, max_sample, self.cluster_size).contiguous()

        # Fused VLAD aggregation: B x N x D, B x N x K -> B x D x K
        x_reshaped = x.view(B, max_sample, self.feature_size).contiguous()
        vlad = triton_vlad_fused(x_reshaped, assignment, self.clusters2)

        # L2 intra norm over feature dimension D
        vlad = torch.nn.functional.normalize(vlad, p=2.0, dim=1)

        # Flatten + L2 norm over descriptor dimension
        vlad = vlad.reshape(B, self.cluster_size * self.feature_size)
        vlad = torch.nn.functional.normalize(vlad, p=2.0, dim=1)

        return vlad
