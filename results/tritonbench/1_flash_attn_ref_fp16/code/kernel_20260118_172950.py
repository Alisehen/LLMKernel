import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ],
    key=['seq_len', 'head_dim'],
)
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qbh, stride_qm, stride_qk,
    stride_kbh, stride_kn, stride_kk,
    stride_vbh, stride_vn, stride_vk,
    stride_obh, stride_om, stride_ok,
    seq_len, head_dim,
    scale,
    num_blocks_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Base pointers for this batch and head (combined)
    q_base = Q_ptr + pid_bh * stride_qbh
    k_base = K_ptr + pid_bh * stride_kbh
    v_base = V_ptr + pid_bh * stride_vbh
    o_base = O_ptr + pid_bh * stride_obh
    
    # Initialize accumulators
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    
    # Mask for valid query positions
    mask_m = offs_m < seq_len
    
    # Load Q block once - shape (BLOCK_M, BLOCK_DMODEL)
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = mask_m[:, None] & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Iterate over K, V blocks
    for block_n in range(num_blocks_n):
        start_n = block_n * BLOCK_N
        curr_offs_n = start_n + offs_n
        mask_n = curr_offs_n < seq_len
        
        # Load K block - shape (BLOCK_N, BLOCK_DMODEL)
        k_ptrs = k_base + curr_offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = mask_n[:, None] & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        
        # Compute QK^T: (BLOCK_M, BLOCK_DMODEL) @ (BLOCK_DMODEL, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, tl.trans(k), allow_tf32=True) * scale
        
        # Mask out-of-bounds positions with large negative value
        qk_mask = mask_m[:, None] & mask_n[None, :]
        qk = tl.where(qk_mask, qk, -1e9)
        
        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Prevent exp overflow by clamping
        alpha = tl.exp(tl.minimum(m_i - m_new, 0.0))
        p = tl.exp(qk - m_new[:, None])
        
        # Mask p for invalid positions
        p = tl.where(qk_mask, p, 0.0)
        
        l_new = alpha * l_i + tl.sum(p, axis=1)
        
        # Load V block - shape (BLOCK_N, BLOCK_DMODEL)
        v_ptrs = v_base + curr_offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = mask_n[:, None] & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        
        # Update accumulator
        pv = tl.dot(p.to(tl.float32), v, allow_tf32=True)
        acc = acc * alpha[:, None] + pv
        
        m_i = m_new
        l_i = l_new
    
    # Final normalization - avoid division by zero
    l_i = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_i[:, None]
    
    # Store output
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = mask_m[:, None] & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, acc.to(tl.float16), mask=o_mask)


def flash_attention(Q, K, V):
    batch, n_heads, seq_len, head_dim = Q.shape
    
    # Output tensor
    O = torch.empty_like(Q)
    
    # Scale factor
    scale = 1.0 / math.sqrt(head_dim)
    
    # BLOCK_DMODEL must be power of 2 and >= head_dim
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    
    # Reshape to combine batch and heads for more parallelism
    Q_flat = Q.view(batch * n_heads, seq_len, head_dim).contiguous()
    K_flat = K.view(batch * n_heads, seq_len, head_dim).contiguous()
    V_flat = V.view(batch * n_heads, seq_len, head_dim).contiguous()
    O_flat = O.view(batch * n_heads, seq_len, head_dim)
    
    # Compute num_blocks_n outside kernel
    # We need to pass this as a runtime value
    
    # Grid: (num_query_blocks, batch * n_heads)
    def grid(META):
        return (triton.cdiv(seq_len, META['BLOCK_M']), batch * n_heads)
    
    # Compute max num_blocks_n for the largest BLOCK_N in configs
    max_block_n = 64  # max BLOCK_N in configs
    num_blocks_n = (seq_len + max_block_n - 1) // max_block_n
    
    flash_attention_kernel[grid](
        Q_flat, K_flat, V_flat, O_flat,
        Q_flat.stride(0), Q_flat.stride(1), Q_flat.stride(2),
        K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
        V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
        O_flat.stride(0), O_flat.stride(1), O_flat.stride(2),
        seq_len, head_dim,
        scale,
        num_blocks_n,
        BLOCK_DMODEL=BLOCK_DMODEL,
    )
    
    return O


class ModelNew(nn.Module):
    """
    Flash Attention implementation using Triton with optimized parallelism
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = False) -> torch.Tensor:
        return flash_attention(Q, K, V)
