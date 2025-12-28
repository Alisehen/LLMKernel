import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------
# Triton Kernels
# ---------------------------

@triton.jit
def linear_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Matrix multiply: C[M, N] = A[M, K] @ B[K, N]
    A is (M x K), B is (K x N), C is (M x N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_mask = (offs_k[None, :] + k) < K  # (1, BLOCK_K)

        # Load A tile: (BLOCK_M, BLOCK_K)
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )

        # Load B tile: (BLOCK_K, BLOCK_N)
        b = tl.load(
            b_ptrs,
            mask=k_mask.T & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_im, stride_in,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    """
    Row-wise softmax over N columns for an MxN matrix.
    Softmax is applied along dim=1 (columns).
    Assumes N <= BLOCK_N.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    row_mask = row < M
    col_mask = offs < N
    mask = row_mask & col_mask

    in_ptrs = input_ptr + row * stride_im + offs * stride_in
    x = tl.load(in_ptrs, mask=mask, other=-float("inf"))

    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0) + 1e-9
    out = exp_x / denom

    out_ptrs = output_ptr + row * stride_om + offs * stride_on
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def vlad_fused_kernel(
    x_ptr,           # B x N x D
    assign_ptr,      # B x N x K
    c2_ptr,          # D x K
    out_ptr,         # B x D x K
    B, N, D, K,
    stride_xb, stride_xn, stride_xd,
    stride_ab, stride_an, stride_ak,
    stride_c2d, stride_c2k,
    stride_ob, stride_od, stride_ok,
    BLOCK_M: tl.constexpr,  # tile over D
    BLOCK_N: tl.constexpr,  # tile over K
    BLOCK_K: tl.constexpr,  # reduction over N
):
    """
    Fused VLAD aggregation:

    For each b,d,k:
      vlad[b,d,k] = sum_n assignment[b,n,k] * x[b,n,d]
                    - clusters2[d,k] * sum_n assignment[b,n,k]

    x:        [B, N, D]
    assign:   [B, N, K]
    clusters2 (c2): [D, K]
    out:      [B, D, K]
    """
    pid_m = tl.program_id(0)  # over D tiles
    pid_n = tl.program_id(1)  # over K tiles
    pid_b = tl.program_id(2)  # over batch

    b = pid_b

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # D indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # K indices

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_sum_assign = tl.zeros((BLOCK_N,), dtype=tl.float32)

    k = 0
    while k < N:
        offs_k = tl.arange(0, BLOCK_K)
        n_idx = k + offs_k               # N indices
        n_mask = n_idx < N               # (BLOCK_K,)

        # x tile: shape (BLOCK_M, BLOCK_K), indices [d, n]
        x_ptrs = x_ptr + (
            b * stride_xb
            + offs_m[:, None] * stride_xd
            + n_idx[None, :] * stride_xn
        )
        x_mask = (offs_m[:, None] < D) & n_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # assignment tile: shape (BLOCK_K, BLOCK_N), indices [n, k]
        a_ptrs = assign_ptr + (
            b * stride_ab
            + n_idx[:, None] * stride_an
            + offs_n[None, :] * stride_ak
        )
        a_mask = n_mask[:, None] & (offs_n[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        acc += tl.dot(x_tile, a_tile)

        partial_sum = tl.sum(a_tile, axis=0)  # (BLOCK_N,)
        acc_sum_assign += partial_sum

        k += BLOCK_K

    # Load clusters2 tile: [D, K]
    c2_ptrs = c2_ptr + offs_m[:, None] * stride_c2d + offs_n[None, :] * stride_c2k
    c2 = tl.load(
        c2_ptrs,
        mask=(offs_m[:, None] < D) & (offs_n[None, :] < K),
        other=0.0,
    )

    # Broadcast acc_sum_assign over D dimension
    a_sum = acc_sum_assign[None, :]  # (1, BLOCK_N)
    out_tile = acc - c2 * a_sum

    out_ptrs = out_ptr + (
        b * stride_ob
        + offs_m[:, None] * stride_od
        + offs_n[None, :] * stride_ok
    )
    tl.store(
        out_ptrs,
        out_tile,
        mask=(offs_m[:, None] < D) & (offs_n[None, :] < K) & (b < B),
    )


@triton.jit
def l2norm_rows_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps,
    BLOCK_N: tl.constexpr,
):
    """
    L2-normalize each row of an MxN matrix along N.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    row_mask = row < M

    # First pass: compute squared-norm (scalar accumulator)
    sum_sq = 0.0

    col_start = 0
    while col_start < N:
        col_idx = col_start + offs
        mask = row_mask & (col_idx < N)
        x_vals = tl.load(
            x_ptr + row * stride_xm + col_idx * stride_xn,
            mask=mask,
            other=0.0,
        )
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
        col_start += BLOCK_N

    inv_norm = 1.0 / tl.sqrt(sum_sq + eps)

    # Second pass: normalize and store
    col_start = 0
    while col_start < N:
        col_idx = col_start + offs
        mask = row_mask & (col_idx < N)
        x_vals = tl.load(
            x_ptr + row * stride_xm + col_idx * stride_xn,
            mask=mask,
            other=0.0,
        )
        y_vals = x_vals * inv_norm
        tl.store(
            y_ptr + row * stride_ym + col_idx * stride_yn,
            y_vals,
            mask=mask,
        )
        col_start += BLOCK_N


# ---------------------------
# Python Wrappers
# ---------------------------

def triton_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x: [M, K]
    weight: [K, N]
    returns: [M, N] = x @ weight
    """
    assert x.is_cuda and weight.is_cuda
    M, K = x.shape
    K_w, N = weight.shape
    assert K == K_w

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32

    def grid(META):
        return (
            max(1, triton.cdiv(M, META["BLOCK_M"])),
            max(1, triton.cdiv(N, META["BLOCK_N"])),
        )

    linear_kernel[grid](
        x_contig, w_contig, out,
        M, N, K,
        x_contig.stride(0), x_contig.stride(1),
        w_contig.stride(0), w_contig.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


def triton_softmax_dim1(x: torch.Tensor) -> torch.Tensor:
    """
    x: [M, N]
    softmax along dim=1
    Assumes N <= 128.
    """
    assert x.is_cuda
    M, N = x.shape
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)

    BLOCK_N = 128

    def grid(META):
        return (max(1, M),)

    softmax_kernel[grid](
        x_contig, out,
        M, N,
        x_contig.stride(0), x_contig.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return out


def triton_fused_vlad(
    x: torch.Tensor,          # [B, N, D]
    assignment: torch.Tensor, # [B, N, K]
    clusters2: torch.Tensor,  # [1, D, K]
) -> torch.Tensor:
    """
    Compute VLAD aggregation fused with cluster centroids subtraction.
    Returns vlad: [B, D, K]
    """
    assert x.is_cuda and assignment.is_cuda and clusters2.is_cuda

    B, N, D = x.shape
    B_a, N_a, K = assignment.shape
    assert B == B_a and N == N_a
    assert clusters2.shape[0] == 1 and clusters2.shape[1] == D and clusters2.shape[2] == K

    x_c = x.contiguous()
    a_c = assignment.contiguous()
    c2_2d = clusters2.view(D, K).contiguous()

    out = torch.empty((B, D, K), device=x.device, dtype=x.dtype)

    BLOCK_M = 64  # over D
    BLOCK_N = 32  # over K
    BLOCK_K = 32  # over N

    def grid(META):
        return (
            max(1, triton.cdiv(D, META["BLOCK_M"])),
            max(1, triton.cdiv(K, META["BLOCK_N"])),
            max(1, B),
        )

    vlad_fused_kernel[grid](
        x_c, a_c, c2_2d, out,
        B, N, D, K,
        x_c.stride(0), x_c.stride(1), x_c.stride(2),
        a_c.stride(0), a_c.stride(1), a_c.stride(2),
        c2_2d.stride(0), c2_2d.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


def triton_l2norm_rows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    L2-normalize each row of 2D tensor x along dim=1.
    Returns new tensor (not in-place).
    """
    assert x.is_cuda
    M, N = x.shape
    x_c = x.contiguous()
    out = torch.empty_like(x_c)

    BLOCK_N = 128

    def grid(META):
        return (max(1, M),)

    l2norm_rows_kernel[grid](
        x_c, out,
        M, N,
        x_c.stride(0), x_c.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return out


# ---------------------------
# Optimized Model
# ---------------------------

class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = 1.0 / math.sqrt(feature_size)
        clusters = cluster_size + ghost_clusters

        # (D x (K+G))
        self.clusters = nn.Parameter(
            init_sc * torch.randn(feature_size, clusters)
        )
        self.batch_norm = nn.BatchNorm1d(clusters)
        # (1 x D x K)
        self.clusters2 = nn.Parameter(
            init_sc * torch.randn(1, feature_size, cluster_size)
        )
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        # x: B x N x D
        B, N, D = x.shape
        assert D == self.feature_size

        x_flat = x.view(-1, self.feature_size)  # (B*N) x D

        if x_flat.device != self.clusters.device:
            msg = f"x.device {x_flat.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # Linear assignment: (BN x D) @ (D x (K+G)) -> BN x (K+G)
        assignment = triton_linear(x_flat, self.clusters)

        # BatchNorm over features (clusters)
        assignment = self.batch_norm(assignment)

        # Softmax along cluster dimension
        assignment = triton_softmax_dim1(assignment)

        # Remove ghost clusters, keep only real clusters K
        assignment = assignment[:, :self.cluster_size]  # BN x K

        # Reshape to B x N x K
        assignment = assignment.view(B, N, self.cluster_size)

        # Fused VLAD: computes both aggregation and centroid subtraction
        # Returns vlad: B x D x K (already aggregated and centered)
        vlad = triton_fused_vlad(x, assignment, self.clusters2)

        # L2 intra norm over D (per (b,k) vector)
        # Implemented as: permute to B x K x D, normalize along D, permute back
        vlad_perm = vlad.permute(0, 2, 1).contiguous()  # B x K x D
        BK, D_ = vlad_perm.shape[0] * vlad_perm.shape[1], vlad_perm.shape[2]
        vlad_2d = vlad_perm.view(BK, D_)
        vlad_2d = triton_l2norm_rows(vlad_2d)
        vlad_perm = vlad_2d.view(B, self.cluster_size, self.feature_size)
        vlad = vlad_perm.permute(0, 2, 1).contiguous()  # B x D x K

        # Flattening + global L2 norm (per sample)
        vlad = vlad.reshape(B, self.cluster_size * self.feature_size)
        vlad = triton_l2norm_rows(vlad)

        return vlad  # B x DK
