# <complete ModelNew code with optimized Triton kernels>

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_D": 64},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_D": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_D": 64},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["S", "D"],
)
@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    BATCH, S, D,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Generic FlashAttention-style fused scaled dot-product attention kernel.

    Computes for each batch-head b:

        O[b, m, d] = sum_n softmax_n( Q[b, m, :] @ K[b, n, :].T * scale ) * V[b, n, d]

    Shapes (flattened over batch*heads):
        Q, K, V: [BATCH, S, D]  (D can be large, e.g., 1024)
        O:       [BATCH, S, D]  (float32 accumulator, cast to fp16 outside)
    """
    pid_m = tl.program_id(axis=0)  # block index along sequence dimension (queries)
    pid_b = tl.program_id(axis=1)  # batch*head index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S

    # Base pointers for this batch-head
    Q_batch = Q_ptr + pid_b * stride_qb
    K_batch = K_ptr + pid_b * stride_kb
    V_batch = V_ptr + pid_b * stride_vb
    O_batch = O_ptr + pid_b * stride_ob

    # Online softmax statistics per row (query)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    start_n = 0
    while start_n < S:
        # Keys/values indices for this block
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S

        # ---- Compute QK^T block: [BLOCK_M, BLOCK_N] ----
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        start_d = 0
        while start_d < D:
            offs_d = start_d + tl.arange(0, BLOCK_D)
            d_mask = offs_d < D

            # Load Q block: [BLOCK_M, BLOCK_D]
            q_ptrs = Q_batch + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

            # Load K block: [BLOCK_N, BLOCK_D]
            k_ptrs = K_batch + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

            # Accumulate partial dot-product over D dimension
            qk += tl.dot(q, tl.trans(k))

            start_d += BLOCK_D

        # Scale and apply causal / bounds mask (only bounds here)
        qk = qk * scale
        qk = tl.where(m_mask[:, None] & n_mask[None, :], qk, -float("inf"))

        # ---- Online softmax over keys (N dimension) ----
        current_max = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, current_max)

        logits_shifted = qk - m_i_new[:, None]
        p = tl.exp(logits_shifted)

        has_prev = l_i > 0
        exp_scale = tl.where(has_prev, tl.exp(m_i - m_i_new), 0.0)
        l_i_scaled = l_i * exp_scale
        p_sum = tl.sum(p, axis=1)
        l_i_new = l_i_scaled + p_sum
        inv_l_new = 1.0 / l_i_new

        alpha = tl.where(has_prev, l_i_scaled * inv_l_new, 0.0)
        beta = inv_l_new

        # ---- Update output accumulator O in blocks of D ----
        start_dv = 0
        while start_dv < D:
            offs_dv = start_dv + tl.arange(0, BLOCK_D)
            dv_mask = offs_dv < D

            # Load V block: [BLOCK_N, BLOCK_D]
            v_ptrs = V_batch + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=n_mask[:, None] & dv_mask[None, :], other=0.0)

            # Compute p @ V_block: [BLOCK_M, BLOCK_D]
            pv = tl.dot(p.to(tl.float16), v)  # accumulate in fp32

            # Load previous accumulator if it exists, else treat as 0
            o_ptrs = O_batch + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od
            acc_prev = tl.load(
                o_ptrs,
                mask=m_mask[:, None] & dv_mask[None, :] & has_prev[:, None],
                other=0.0,
            )

            acc = acc_prev * alpha[:, None] + pv * beta[:, None]

            tl.store(o_ptrs, acc, mask=m_mask[:, None] & dv_mask[None, :])

            start_dv += BLOCK_D

        # Update running softmax stats
        m_i = m_i_new
        l_i = l_i_new

        start_n += BLOCK_N


def triton_scaled_dot_product_attention(Q: torch.Tensor,
                                        K: torch.Tensor,
                                        V: torch.Tensor) -> torch.Tensor:
    """
    Fused Triton implementation of scaled dot-product attention:

        attn = softmax(Q @ K^T / sqrt(d_k))
        out  = attn @ V

    Inputs:
        Q, K, V: [B, H, S, D], dtype=float16, CUDA tensors
    Output:
        out: [B, H, S, D], dtype=float16

    Supports arbitrary head_dim D (e.g., 1024) via tiling over D.
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Inputs must be CUDA tensors"
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16
    assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"

    B, H, S, D = Q.shape
    BH = B * H

    # Flatten [B, H] -> batch dimension for the kernel
    Q_flat = Q.contiguous().view(BH, S, D)
    K_flat = K.contiguous().view(BH, S, D)
    V_flat = V.contiguous().view(BH, S, D)

    # Accumulator/output buffer in fp32 for better numerical stability
    O_accum = torch.empty((BH, S, D), device=Q.device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(D)

    def grid(meta):
        return (
            triton.cdiv(S, meta["BLOCK_M"]),
            BH,
        )

    flash_attn_fwd_kernel[grid](
        Q_flat, K_flat, V_flat, O_accum,
        BH, S, D,
        Q_flat.stride(0), Q_flat.stride(1), Q_flat.stride(2),
        K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
        V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
        O_accum.stride(0), O_accum.stride(1), O_accum.stride(2),
        scale,
    )

    O_flat = O_accum.to(torch.float16)
    out = O_flat.view(B, H, S, D)
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_scaled_dot_product_attention(Q, K, V)
