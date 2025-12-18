import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline (conservative) config: low register pressure, good occupancy
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        # Larger N tile for better memory throughput on long sequences
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        # Higher warp count for compute-bound regimes when register pressure allows
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ],
    key=["T", "HEAD_DIM"],
)
@triton.jit
def causal_self_attn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qb,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vd,
    stride_ob,
    stride_ot,
    stride_od,
    T,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused FlashAttention-style causal self-attention forward kernel.

    Layout: Q, K, V, O all as [BH, T, D] with D contiguous (HEAD_DIM == D).
    Each program computes a [BLOCK_M, HEAD_DIM] tile of O for a single
    batch*head index, streaming over K/V in BLOCK_N chunks with online softmax.
    """
    pid_m = tl.program_id(0)  # sequence tile (queries)
    pid_bh = tl.program_id(1)  # batch*head index

    # -------- Offsets & masks --------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, HEAD_DIM)                   # [HEAD_DIM]
    mask_m = offs_m < T                               # [BLOCK_M]
    mask_d = offs_d < HEAD_DIM                        # always true but keeps code generic

    # Base pointers for this (batch, head) slice
    q_bh_ptr = q_ptr + pid_bh * stride_qb
    k_bh_ptr = k_ptr + pid_bh * stride_kb
    v_bh_ptr = v_ptr + pid_bh * stride_vb
    o_bh_ptr = o_ptr + pid_bh * stride_ob

    # -------- Load Q block [BLOCK_M, HEAD_DIM] --------
    q_ptrs = q_bh_ptr + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Online softmax state per row
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # -------- Loop over K/V blocks along sequence dimension --------
    offs_n_init = tl.arange(0, BLOCK_N)
    start_n = 0

    while start_n < T:
        offs_n = start_n + offs_n_init           # [BLOCK_N]
        mask_n = offs_n < T                      # [BLOCK_N]

        # ---- Load K block: [HEAD_DIM, BLOCK_N] ----
        k_ptrs = (
            k_bh_ptr
            + offs_n[None, :] * stride_kt
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)

        # ---- Attention scores: [BLOCK_M, BLOCK_N] ----
        scores = tl.dot(q, k, out_dtype=tl.float32)
        scores = scores * scale

        # ---- Causal + bounds mask ----
        causal_mask = offs_n[None, :] <= offs_m[:, None]
        full_mask = causal_mask & mask_m[:, None] & mask_n[None, :]

        scores = tl.where(full_mask, scores, -float("inf"))

        # ---- Online softmax update ----
        current_max = tl.max(scores, axis=1)          # [BLOCK_M]
        m_i_new = tl.maximum(m_i, current_max)
        alpha = tl.exp(m_i - m_i_new)                 # [BLOCK_M]

        scores = scores - m_i_new[:, None]
        p = tl.exp(scores)                            # [BLOCK_M, BLOCK_N]
        p = tl.where(full_mask, p, 0.0)
        p_sum = tl.sum(p, axis=1)                     # [BLOCK_M]
        l_i = l_i * alpha + p_sum

        # ---- Load V block: [BLOCK_N, HEAD_DIM] ----
        v_ptrs = (
            v_bh_ptr
            + offs_n[:, None] * stride_vt
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # ---- Accumulate output: acc = acc*alpha + p @ V ----
        contrib = tl.dot(p, v, out_dtype=tl.float32)  # [BLOCK_M, HEAD_DIM]
        acc = acc * alpha[:, None] + contrib

        m_i = m_i_new
        start_n += BLOCK_N

    # -------- Final normalization: acc / l_i --------
    out = acc / l_i[:, None]

    # -------- Store output tile --------
    o_ptrs = o_bh_ptr + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


def fused_causal_attention(q, k, v):
    """
    Fused causal self-attention using Triton FlashAttention-style kernel.

    Args:
      q, k, v: [B, n_head, T, head_dim] on CUDA, last dim contiguous.
               Recommended dtypes: torch.float16 or torch.bfloat16.

    Returns:
      y: [B, n_head, T, head_dim]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.shape == k.shape == v.shape
    B, n_head, T, D = q.shape
    BH = B * n_head

    # Flatten to [BH, T, D] (D contiguous)
    q_flat = q.view(BH, T, D)
    k_flat = k.view(BH, T, D)
    v_flat = v.view(BH, T, D)

    o_flat = torch.empty_like(q_flat)

    stride_qb, stride_qt, stride_qd = q_flat.stride()
    stride_kb, stride_kt, stride_kd = k_flat.stride()
    stride_vb, stride_vt, stride_vd = v_flat.stride()
    stride_ob, stride_ot, stride_od = o_flat.stride()

    # Scale factor for dot-product attention
    scale = 1.0 / (D ** 0.5)

    # Kernel grid: one block per query tile per (batch*head)
    def grid(meta):
        return (triton.cdiv(T, meta["BLOCK_M"]), BH)

    causal_self_attn_fwd_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        o_flat,
        stride_qb,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vt,
        stride_vd,
        stride_ob,
        stride_ot,
        stride_od,
        T,
        scale,
        HEAD_DIM=D,
    )

    return o_flat.view(B, n_head, T, D)


class ModelNew(nn.Module):
    """
    Triton-accelerated masked multi-head self-attention block.

    The attention core (QK^T + causal mask + softmax + @V) is fused
    into a single Triton kernel (FlashAttention-style) when attention
    dropout is inactive. When attention dropout is active during
    training, it falls back to the exact PyTorch implementation.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Dropouts
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # Causal mask for fallback PyTorch path
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen))
            .view(1, 1, max_seqlen, max_seqlen),
        )

    def forward(self, x):
        B, T, C = x.size()

        # If attention dropout is active during training, use the exact PyTorch path
        if self.training and self.attn_dropout.p > 0.0:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            return y

        # Fused Triton path (attention dropout inactive)
        qkv = self.c_attn(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=2)

        # [B, T, nh, hd] -> [B, nh, T, hd]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()

        if not x.is_cuda:
            # CPU fallback: use unfused PyTorch implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = torch.softmax(att, dim=-1)
            y = att @ v
        else:
            # Triton fused FlashAttention core
            y = fused_causal_attention(q, k, v)

        # Merge heads and apply output projection + residual dropout
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
