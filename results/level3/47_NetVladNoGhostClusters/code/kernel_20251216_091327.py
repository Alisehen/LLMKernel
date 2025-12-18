# <corrected code>

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


# ---------------------------------------------------------------------------
# High-performance GEMM: C = A @ B
# A: (M, K), B: (K, N), C: (M, N)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Computes C = A @ B
    A: (M, K)
    B: (K, N)
    C: (M, N)
    All in row-major with given strides.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak
        b_ptrs = b_ptr + k_idxs[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k_idxs[None, :] < K)
        b_mask = (k_idxs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ---------------------------------------------------------------------------
# High-performance VLAD-style batched matmul:
# For each batch b:
#   A[b]: (N, K)  assignment_3d  (N = #points, K = #clusters)
#   X[b]: (N, D)  feature vectors
#   OUT[b]: (K, D) with
#       OUT[b, k, d] = sum_n A[b, n, k] * X[b, n, d]
# This computes A[b]^T @ X[b] without explicitly transposing A.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=['N_points', 'K_clusters', 'D_features'],
)
@triton.jit
def vlad_batched_matmul_kernel(
    a_ptr, x_ptr, out_ptr,
    B, N_points, K_clusters, D_features,
    stride_ab, stride_an, stride_ak,       # A: (B, N, K)
    stride_xb, stride_xn, stride_xd,       # X: (B, N, D)
    stride_ob, stride_ok, stride_od,       # OUT: (B, K, D)
    BLOCK_M: tl.constexpr,                 # tile size along K (clusters)
    BLOCK_N: tl.constexpr,                 # tile size along D (features)
    BLOCK_K: tl.constexpr,                 # reduction tile along N (points)
):
    """
    Batched VLAD-style matmul:
      For each batch b: OUT[b] = A[b]^T @ X[b]
    A[b]: (N_points, K_clusters)
    X[b]: (N_points, D_features)
    OUT[b]: (K_clusters, D_features)
    """

    pid_m = tl.program_id(0)  # over K_clusters
    pid_n = tl.program_id(1)  # over D_features
    pid_b = tl.program_id(2)  # over batches

    offs_k = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # cluster indices
    offs_d = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # feature indices
    offs_n = tl.arange(0, BLOCK_K)                     # reduction over points

    a_batch_ptr = a_ptr + pid_b * stride_ab
    x_batch_ptr = x_ptr + pid_b * stride_xb
    out_batch_ptr = out_ptr + pid_b * stride_ob

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over N_points (reduction dimension)
    for n0 in range(0, N_points, BLOCK_K):
        n_idxs = n0 + offs_n

        # A_chunk shape: [BLOCK_M, BLOCK_K] corresponding to (K, N_chunk)
        a_ptrs = a_batch_ptr \
            + offs_k[:, None] * stride_ak \
            + n_idxs[None, :] * stride_an

        # X_chunk shape: [BLOCK_K, BLOCK_N] corresponding to (N_chunk, D)
        x_ptrs = x_batch_ptr \
            + n_idxs[:, None] * stride_xn \
            + offs_d[None, :] * stride_xd

        a_mask = (offs_k[:, None] < K_clusters) & (n_idxs[None, :] < N_points)
        x_mask = (n_idxs[:, None] < N_points) & (offs_d[None, :] < D_features)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(a, x, allow_tf32=True)

    out_ptrs = out_batch_ptr \
        + offs_k[:, None] * stride_ok \
        + offs_d[None, :] * stride_od
    out_mask = (offs_k[:, None] < K_clusters) & (offs_d[None, :] < D_features)
    tl.store(out_ptrs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Wrapper: single matmul
# ---------------------------------------------------------------------------

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute a @ b using Triton.
    a: (M, K)
    b: (K, N)
    returns: (M, N)
    """
    assert a.ndim == 2 and b.ndim == 2, "triton_matmul expects 2D tensors"
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Incompatible matmul shapes"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ---------------------------------------------------------------------------
# Wrapper: VLAD-style batched matmul without explicit transpose:
#   (B, N, K) and (B, N, D) -> (B, K, D)
# ---------------------------------------------------------------------------

def triton_batched_matmul(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Batched VLAD-style matmul using Triton.
    a: (B, N, K)  assignment (points x clusters)
    x: (B, N, D)  features (points x dim)
    returns: (B, K, D)  aggregated per-cluster features
    """
    assert a.ndim == 3 and x.ndim == 3, "triton_batched_matmul expects 3D tensors"
    B, N_points, K_clusters = a.shape
    B2, N2, D_features = x.shape
    assert B == B2 and N_points == N2, "Incompatible batched matmul shapes"

    out = torch.empty((B, K_clusters, D_features), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (
            triton.cdiv(K_clusters, meta['BLOCK_M']),
            triton.cdiv(D_features, meta['BLOCK_N']),
            B,
        )

    vlad_batched_matmul_kernel[grid](
        a, x, out,
        B, N_points, K_clusters, D_features,
        a.stride(0), a.stride(1), a.stride(2),
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
    )
    return out


# ---------------------------------------------------------------------------
# NetVLAD-like model using optimized Triton kernels
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Triton-optimized version of the NetVLAD-like module.

    Heavy matrix multiplications are offloaded to Triton kernels:
      - assignment = x @ clusters
      - vlad = assignment^T @ x (batched, implemented without explicit transpose)
    The rest (BatchNorm, softmax, reductions, normalizations)
    is kept in PyTorch for correctness and training compatibility.
    """

    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = 1.0 / math.sqrt(feature_size)
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): B x N x D

        Returns:
            (torch.Tensor): B x (D * K)
        """
        B, max_sample, D = x.shape
        assert D == self.feature_size, "Input feature_size mismatch"

        # B x N x D -> (B*N) x D
        x_flat = x.view(-1, self.feature_size)

        if x_flat.device != self.clusters.device:
            msg = f"x.device {x_flat.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = triton_matmul(x_flat, self.clusters)

        # BatchNorm1d over the (K+G) features
        assignment = self.batch_norm(assignment)

        # Softmax over cluster dimension
        assignment = torch.nn.functional.softmax(assignment, dim=1)  # BN x (K+G)

        # Remove ghost clusters: BN x (K+G) -> BN x K
        assignment = assignment[:, :self.cluster_size]

        # BN x K -> B x N x K
        assignment_3d = assignment.view(B, max_sample, self.cluster_size)

        # B x N x K -> B x 1 x K (sum over N)
        a_sum = torch.sum(assignment_3d, dim=1, keepdim=True)

        # B x 1 x K * 1 x D x K -> B x D x K (broadcast over batch and N)
        a = a_sum * self.clusters2

        # BN x D -> B x N x D
        x_3d = x_flat.view(B, max_sample, self.feature_size)  # B x N x D

        # (B x N x K) and (B x N x D) -> B x K x D via Triton batched matmul
        vlad = triton_batched_matmul(assignment_3d, x_3d)  # B x K x D

        # B x K x D -> B x D x K
        vlad = vlad.transpose(1, 2)  # B x D x K

        # Residuals: subtract cluster centers
        vlad = vlad - a  # B x D x K

        # L2 intra norm over feature dimension (dim=1)
        vlad = torch.nn.functional.normalize(vlad)

        # Flatten to B x (D*K)
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)

        # Global L2 norm over descriptor dimension (dim=1)
        vlad = torch.nn.functional.normalize(vlad)

        return vlad
