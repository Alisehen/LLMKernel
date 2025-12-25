# <optimized Triton code>

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    ],
    key=["N_CTX"],
)
@triton.jit
def flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qm, stride_qk,
    stride_kb, stride_kn, stride_kk,
    stride_vb, stride_vk, stride_vn,
    stride_ob, stride_om, stride_ok,
    sm_scale,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,  # queries per program (tile in sequence dim)
    BLOCK_N: tl.constexpr,  # keys per tile
    BLOCK_D: tl.constexpr,  # head dim tile (power-of-2, >= D_HEAD)
):
    # -------------------------------------------------------------------------
    # program ids: grid covers OUTPUT tensor [BH, S, D]
    #   pid_m  -> range over S in tiles of BLOCK_M (rows / queries)
    #   pid_bh -> range over BH (batch * heads)
    # -------------------------------------------------------------------------
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # row (query) indices this program will compute
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # head-dim indices
    offs_d = tl.arange(0, BLOCK_D)

    # shared mask for head dimension (reused by all fused ops)
    mask_d = offs_d < D_HEAD
    # shared mask for query rows (reused by all fused ops)
    mask_m = offs_m < N_CTX

    # -------------------------------------------------------------------------
    # Load Q tile for this (bh, block of queries)
    #   q_ptrs shape: [BLOCK_M, BLOCK_D]
    # -------------------------------------------------------------------------
    q_ptrs = (
        q_ptr
        + pid_bh * stride_qb
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q = tl.load(
        q_ptrs,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    q_dtype = q.dtype
    q = q.to(tl.float32)

    # -------------------------------------------------------------------------
    # Online softmax stats and output accumulator (all in fp32)
    # m_i: running max per row
    # l_i: running sum of exp scores per row
    # acc: running output [BLOCK_M, BLOCK_D]
    # -------------------------------------------------------------------------
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # Loop over key/value tiles along sequence dimension
    # -------------------------------------------------------------------------
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < N_CTX  # shared K/V mask for this tile

        # ---------------------------------------------------------------------
        # Load K tile: shape [BLOCK_D, BLOCK_N]
        # ---------------------------------------------------------------------
        k_ptrs = (
            k_ptr
            + pid_bh * stride_kb
            + offs_d[:, None] * stride_kk
            + offs_n[None, :] * stride_kn
        )
        k = tl.load(
            k_ptrs,
            mask=mask_d[:, None] & kv_mask[None, :],
            other=0.0,
        )
        k = k.to(tl.float32)

        # ---------------------------------------------------------------------
        # Compute QK^T for this tile: [BLOCK_M, BLOCK_N]
        # q: [BLOCK_M, BLOCK_D], k: [BLOCK_D, BLOCK_N]
        # ---------------------------------------------------------------------
        qk = tl.dot(q, k, allow_tf32=True)
        qk = qk * sm_scale

        # Mask out-of-range keys to -inf so they don't affect softmax
        qk = tl.where(kv_mask[None, :], qk, -float("inf"))

        # ---------------------------------------------------------------------
        # Online softmax update
        # ---------------------------------------------------------------------
        # Tile-wise max over BLOCK_N for each row
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        # Probabilities relative to new global max
        p = tl.exp(qk - m_i_new[:, None])
        p_sum = tl.sum(p, axis=1)

        # Update normalizer
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = l_i * alpha + p_sum

        # ---------------------------------------------------------------------
        # Apply probabilities to V tile
        #   v: [BLOCK_N, BLOCK_D]
        #   p: [BLOCK_M, BLOCK_N]
        #  pv: [BLOCK_M, BLOCK_D]
        # ---------------------------------------------------------------------
        v_ptrs = (
            v_ptr
            + pid_bh * stride_vb
            + offs_n[:, None] * stride_vk
            + offs_d[None, :] * stride_vn
        )
        v = tl.load(
            v_ptrs,
            mask=kv_mask[:, None] & mask_d[None, :],
            other=0.0,
        )
        v = v.to(tl.float32)

        pv = tl.dot(p, v, allow_tf32=True)

        # Combine with previous accumulator
        acc_scale = (l_i * alpha) / l_i_new
        acc = acc * acc_scale[:, None] + pv / l_i_new[:, None]

        # Commit updated stats
        m_i = m_i_new
        l_i = l_i_new

    # -------------------------------------------------------------------------
    # Write output O for this (bh, block of queries)
    # Grid & offsets shared by all fused output ops
    # -------------------------------------------------------------------------
    o_ptrs = (
        o_ptr
        + pid_bh * stride_ob
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_ok
    )
    out = acc.to(q_dtype)
    out_mask = mask_m[:, None] & mask_d[None, :]
    tl.store(o_ptrs, out, mask=out_mask)


def flash_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Fused FlashAttention-style forward:
      out = softmax(Q @ K^T / sqrt(d)) @ V

    Q, K, V: [B, H, S, D]
    returns: [B, H, S, D]
    """
    assert Q.shape == K.shape == V.shape
    B, H, S, D = Q.shape
    BH = B * H

    # Make [BH, S, D] contiguous for optimal memory access
    q = Q.contiguous().view(BH, S, D)
    k = K.contiguous().view(BH, S, D)
    v = V.contiguous().view(BH, S, D)
    out = torch.empty_like(q)

    # Strides in elements
    stride_qb, stride_qm, stride_qk = q.stride()
    stride_kb, stride_kn, stride_kk = k.stride()
    stride_vb, stride_vk, stride_vn = v.stride()
    stride_ob, stride_om, stride_ok = out.stride()

    # Choose BLOCK_D as the smallest power-of-2 >= D, capped at 256
    if D <= 16:
        BLOCK_D = 16
    elif D <= 32:
        BLOCK_D = 32
    elif D <= 64:
        BLOCK_D = 64
    elif D <= 128:
        BLOCK_D = 128
    else:
        BLOCK_D = 256

    sm_scale = 1.0 / math.sqrt(D)

    # Grid covers OUTPUT tensor dimensions [BH, S, D]
    grid = (
        triton.cdiv(S, 64),  # pid_m tiles sequence length; BLOCK_M autotuned but <= 128
        BH,                  # pid_bh tiles batch*heads
    )

    flash_attn_fwd_kernel[grid](
        q, k, v, out,
        stride_qb, stride_qm, stride_qk,
        stride_kb, stride_kn, stride_kk,
        stride_vb, stride_vk, stride_vn,
        stride_ob, stride_om, stride_ok,
        sm_scale,
        N_CTX=S,
        D_HEAD=D,
        BLOCK_D=BLOCK_D,
    )

    return out.view(B, H, S, D)


class ModelNew(nn.Module):
    """
    Triton-optimized scaled dot-product attention:
      softmax(Q @ K^T / sqrt(d)) @ V

    Matches the behavior of the reference PyTorch attention.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return flash_attn_fwd(Q, K, V)
