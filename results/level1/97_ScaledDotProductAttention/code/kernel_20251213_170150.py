# <complete ModelNew code with optimized Triton kernels>
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    ],
    key=["S"],
)
@triton.jit
def sdpa_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_vbh, stride_vm, stride_vk,
    stride_obh, stride_om, stride_ok,
    BH, S,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused scaled dot-product attention:
        O = softmax(Q @ K^T / sqrt(d_k)) @ V

    Operates on flattened [BH, S, D] tensors, where:
        BH = B * H, S = sequence length, D = head dimension (HEAD_DIM).
    Each program instance processes a block of queries (BLOCK_M) for one [b,h] head.
    Softmax over keys is computed in a streaming fashion, avoiding explicit [S, S] storage.
    """
    pid_m = tl.program_id(axis=0)  # block index along sequence length
    pid_bh = tl.program_id(axis=1)  # head index (flattened B*H)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Masks
    m_mask = offs_m < S  # valid query rows

    # Load Q block [BLOCK_M, HEAD_DIM]
    q_ptrs = (
        Q_ptr
        + pid_bh * stride_qbh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float16)

    # Initialize running softmax statistics and output accumulator
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                # running sum exp
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)       # output accumulator

    n_start = 0
    # Iterate over key/value blocks along sequence dimension
    while n_start < S:
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S

        # Load K and V blocks: [BLOCK_N, HEAD_DIM]
        k_ptrs = (
            K_ptr
            + pid_bh * stride_kbh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kk
        )
        v_ptrs = (
            V_ptr
            + pid_bh * stride_vbh
            + offs_n[:, None] * stride_vm
            + offs_d[None, :] * stride_vk
        )

        kv_mask = n_mask[:, None]  # mask for K/V over (N, D)

        k = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float16)
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float16)

        # Compute attention logits: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k))  # fp16->fp32 accumulate
        qk = qk * scale

        # Mask out invalid query/key positions
        qk_mask = m_mask[:, None] & n_mask[None, :]
        qk = tl.where(qk_mask, qk, -float("inf"))

        # Numerically-stable streaming softmax update
        max_logits = tl.max(qk, axis=1)  # [BLOCK_M]
        max_logits = tl.where(m_mask, max_logits, -float("inf"))

        m_new = tl.maximum(m_i, max_logits)
        # Compute exp(logits - m_new) for current block
        p = tl.exp(qk - m_new[:, None])

        # Update l_i (sum of exp)
        exp_m_diff = tl.exp(m_i - m_new)
        l_new = l_i * exp_m_diff + tl.sum(p, axis=1)
        l_new = tl.where(m_mask, l_new, l_i)

        # Factors for combining old accumulator with new contributions
        # acc_new = acc * (l_i * exp(m_i - m_new) / l_new) + (p @ v) / l_new
        inv_l_new = tl.where(m_mask, 1.0 / l_new, 0.0)
        alpha = l_i * exp_m_diff * inv_l_new
        alpha = tl.where(m_mask, alpha, 0.0)

        # Update accumulator: [BLOCK_M, HEAD_DIM]
        pv = tl.dot(p.to(tl.float32), v.to(tl.float32))
        acc = acc * alpha[:, None] + pv * inv_l_new[:, None]

        # Roll forward running stats
        m_i = tl.where(m_mask, m_new, m_i)
        l_i = l_new

        n_start += BLOCK_N

    # Store output O block [BLOCK_M, HEAD_DIM]
    o_ptrs = (
        O_ptr
        + pid_bh * stride_obh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_ok
    )
    o_mask = m_mask[:, None]  # all D positions are valid for valid rows
    tl.store(o_ptrs, acc.to(tl.float16), mask=o_mask)


def triton_scaled_dot_product_attention(Q: torch.Tensor,
                                        K: torch.Tensor,
                                        V: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of scaled dot-product attention:

        attn = softmax(Q @ K^T / sqrt(d_k))
        out  = attn @ V

    Inputs:
        Q, K, V: [B, H, S, D], float16 CUDA tensors
    Output:
        out: [B, H, S, D], float16 CUDA tensor
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Inputs must be CUDA tensors"
    assert Q.dtype == torch.float16 and K.dtype == torch.float16 and V.dtype == torch.float16
    assert Q.shape == K.shape == V.shape

    B, H, S, D = Q.shape
    BH = B * H

    # Flatten batch and heads
    Q_flat = Q.contiguous().view(BH, S, D)
    K_flat = K.contiguous().view(BH, S, D)
    V_flat = V.contiguous().view(BH, S, D)

    O_flat = torch.empty_like(Q_flat)

    scale = 1.0 / math.sqrt(D)

    def grid(meta):
        return (triton.cdiv(S, meta["BLOCK_M"]), BH)

    sdpa_fwd_kernel[grid](
        Q_flat, K_flat, V_flat, O_flat,
        Q_flat.stride(0), Q_flat.stride(1), Q_flat.stride(2),
        K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
        V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
        O_flat.stride(0), O_flat.stride(1), O_flat.stride(2),
        BH, S,
        scale,
        HEAD_DIM=D,
    )

    return O_flat.view(B, H, S, D)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return triton_scaled_dot_product_attention(Q, K, V)
