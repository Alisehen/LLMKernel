import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B
    A: [M, K]
    B: [K, N]
    C: [M, N]
    Row-major for all matrices.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    High-performance linear: y = x @ weight
    x: [M, K]
    weight: [K, N]
    returns y: [M, N]
    """
    assert x.is_cuda and weight.is_cuda, "Triton linear requires CUDA tensors"
    M, K = x.shape
    K_w, N = weight.shape
    assert K == K_w, "Incompatible shapes for matmul"

    if M == 0 or N == 0 or K == 0:
        return x @ weight

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_gemm_kernel[grid](
        x, weight, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=128, BLOCK_N=64, BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return y


@triton.jit
def batched_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Batched GEMM:
    A: [B, M, K]
    B: [B, K, N]
    C: [B, M, N]
    All in row-major within each batch.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    if pid_b >= B:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_batch_ptr = a_ptr + pid_b * stride_ab
    b_batch_ptr = b_ptr + pid_b * stride_bb
    c_batch_ptr = c_ptr + pid_b * stride_cb

    a_ptrs = a_batch_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_batch_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_batch_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_batched_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    High-performance batched matmul: C = A @ B
    A: [B, M, K]
    B: [B, K, N]
    returns C: [B, M, N]
    """
    assert a.is_cuda and b.is_cuda, "Triton batched matmul requires CUDA tensors"
    B, M, K = a.shape
    Bb, Kb, N = b.shape
    assert B == Bb and K == Kb, "Incompatible shapes for batched matmul"

    if B == 0 or M == 0 or N == 0 or K == 0:
        return torch.matmul(a, b)

    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
        B,
    )

    batched_gemm_kernel[grid](
        a, b, c,
        B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_M=32, BLOCK_N=64, BLOCK_K=32,
        num_warps=4,
        num_stages=3,
    )
    return c


@triton.jit
def l2_normalize_2d_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps,
    BLOCK_N: tl.constexpr,
):
    """
    Row-wise L2 normalization for 2D tensor X of shape [M, N].
    Each row i: y[i, :] = x[i, :] / max(||x[i, :]||_2, eps)
    """
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)

    # First pass: compute squared L2 norm
    sum_sq = 0.0
    for n in range(0, N, BLOCK_N):
        cols = n + offs_n
        mask = cols < N
        x = tl.load(
            x_ptr + row * stride_xm + cols * stride_xn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    norm = tl.sqrt(sum_sq + eps)
    inv_norm = 1.0 / norm

    # Second pass: scale and store
    for n in range(0, N, BLOCK_N):
        cols = n + offs_n
        mask = cols < N
        x = tl.load(
            x_ptr + row * stride_xm + cols * stride_xn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        y = x * inv_norm
        tl.store(
            y_ptr + row * stride_ym + cols * stride_yn,
            y,
            mask=mask,
        )


def triton_l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Row-wise L2 normalization for a 2D tensor using Triton.
    Equivalent to torch.nn.functional.normalize(x, p=2, dim=1, eps=eps).
    """
    assert x.is_cuda, "Triton L2 normalization requires CUDA tensor"
    assert x.dim() == 2, "triton_l2_normalize expects a 2D tensor"
    M, N = x.shape

    if M == 0 or N == 0:
        return torch.nn.functional.normalize(x, p=2, dim=1, eps=eps)

    y = torch.empty_like(x)

    grid = lambda META: (M,)

    l2_normalize_2d_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        eps,
        BLOCK_N=128,
        num_warps=2,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated version of NetVLAD-like module.

    Fuses the heavy linear and batched matmul operations into high-performance
    Triton kernels and replaces F.normalize with Triton L2 normalization.
    """

    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(ModelNew, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = 1 / math.sqrt(feature_size)
        clusters = cluster_size + ghost_clusters

        # (D, K+G)
        self.clusters = nn.Parameter(
            init_sc * torch.randn(feature_size, clusters)
        )
        self.batch_norm = nn.BatchNorm1d(clusters)
        # (1, D, K)
        self.clusters2 = nn.Parameter(
            init_sc * torch.randn(1, feature_size, cluster_size)
        )
        self.out_dim = self.cluster_size * feature_size

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): B x N x D

        Returns:
            Tensor: B x (D * K)
        """
        # Ensure contiguous for fast Triton access
        x = x.contiguous()
        B, max_sample, D = x.shape
        assert D == self.feature_size

        # BN x D
        x_flat = x.view(-1, self.feature_size)

        if x_flat.device != self.clusters.device:
            msg = f"x.device {x_flat.device} != cluster.device {self.clusters.device}"
            raise ValueError(msg)

        # BN x (K+G): high-performance Triton GEMM
        assignment = triton_linear(x_flat, self.clusters)

        # BatchNorm over features (K+G)
        assignment = self.batch_norm(assignment)

        # Softmax over clusters
        assignment = torch.nn.functional.softmax(assignment, dim=1)

        # Remove ghost clusters: BN x K
        assignment = assignment[:, :self.cluster_size]

        # B x N x K
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        # B x 1 x K
        a_sum = assignment.sum(dim=1, keepdim=True)
        a = a_sum * self.clusters2  # broadcast over D

        # B x K x N
        assignment_t = assignment.transpose(1, 2).contiguous()

        # B x N x D
        x_reshaped = x_flat.view(-1, max_sample, self.feature_size)

        # vlad = (B x K x N) @ (B x N x D) -> B x K x D
        vlad = triton_batched_matmul(assignment_t, x_reshaped)

        # B x D x K
        vlad = vlad.transpose(1, 2)

        # Subtract cluster centers contribution
        vlad = vlad - a

        # Intra-normalization along feature dimension (dim=1)
        # Original: F.normalize(vlad) with vlad.shape = [B, D, K] and dim=1
        Bv, Dv, Kv = vlad.shape
        vlad_2d = vlad.permute(0, 2, 1).contiguous().view(Bv * Kv, Dv)
        vlad_2d = triton_l2_normalize(vlad_2d, eps=1e-12)
        vlad = vlad_2d.view(Bv, Kv, Dv).permute(0, 2, 1).contiguous()

        # Flatten: B x (D*K)
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)

        # Global L2 normalization over flattened representation
        vlad = triton_l2_normalize(vlad, eps=1e-12)

        return vlad
