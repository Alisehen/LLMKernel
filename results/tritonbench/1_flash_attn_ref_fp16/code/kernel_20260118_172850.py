import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch, n_heads, seq_len, head_dim,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # Program IDs - use 3D grid for better parallelism
    pid_m = tl.program_id(0)  # which block of queries
    pid_h = tl.program_id(1)  # head index
    pid_b = tl.program_id(2)  # batch index
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Base pointers for this batch and head
    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh
    
    # Initialize accumulators
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    
    # Load Q block once - shape (BLOCK_M, BLOCK_DMODEL)
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float16)
    
    # Iterate over K, V blocks
    num_blocks_n = tl.cdiv(seq_len, BLOCK_N)
    for block_n in range(num_blocks_n):
        start_n = block_n * BLOCK_N
        curr_offs_n = start_n + offs_n
        
        # Load K block - shape (BLOCK_N, BLOCK_DMODEL)
        k_ptrs = k_base + curr_offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = (curr_offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float16)
        
        # Compute QK^T: (BLOCK_M, BLOCK_DMODEL) @ (BLOCK_DMODEL, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, tl.trans(k), allow_tf32=True).to(tl.float32) * scale
        
        # Mask out-of-bounds positions
        qk_mask = (offs_m[:, None] < seq_len) & (curr_offs_n[None, :] < seq_len)
        qk = tl.where(qk_mask, qk, float('-inf'))
        
        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        
        # Load V block - shape (BLOCK_N, BLOCK_DMODEL)
        v_ptrs = v_base + curr_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = (curr_offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float16)
        
        # Update accumulator
        p_fp16 = p.to(tl.float16)
        pv = tl.dot(p_fp16, v, allow_tf32=True).to(tl.float32)
        acc = acc * alpha[:, None] + pv
        
        m_i = m_new
        l_i = l_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, acc.to(tl.float16), mask=o_mask)


def flash_attention(Q, K, V):
    batch, n_heads, seq_len, head_dim = Q.shape
    
    # Output tensor
    O = torch.empty_like(Q)
    
    # Scale factor
    scale = 1.0 / math.sqrt(head_dim)
    
    # Block sizes optimized for RTX 4090
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = 64  # Must be >= head_dim
    
    # Ensure BLOCK_DMODEL covers head_dim
    if head_dim > 64:
        BLOCK_DMODEL = 128
    
    # 3D Grid: (num_query_blocks, n_heads, batch)
    grid = (triton.cdiv(seq_len, BLOCK_M), n_heads, batch)
    
    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        batch, n_heads, seq_len, head_dim,
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return O


class ModelNew(nn.Module):
    """
    Flash Attention implementation using Triton with optimized grid layout
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = False) -> torch.Tensor:
        return flash_attention(Q, K, V)
