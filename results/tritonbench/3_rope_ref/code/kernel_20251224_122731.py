import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def rope_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    q_out_ptr, k_out_ptr,
    BH, S, D, HALF_D,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_qo0, stride_qo1, stride_qo2,
    stride_ko0, stride_ko1, stride_ko2,
    stride_cos0, stride_cos1,
    stride_sin0, stride_sin1,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_block_d = tl.program_id(2)

    offs_d = pid_block_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_d = offs_d < HALF_D
    valid = (pid_bh < BH) & (pid_s < S)
    mask = mask_d & valid

    # Pointers for q halves
    q1_ptrs = q_ptr + pid_bh * stride_q0 + pid_s * stride_q1 + offs_d * stride_q2
    q2_ptrs = q_ptr + pid_bh * stride_q0 + pid_s * stride_q1 + (offs_d + HALF_D) * stride_q2

    # Pointers for k halves
    k1_ptrs = k_ptr + pid_bh * stride_k0 + pid_s * stride_k1 + offs_d * stride_k2
    k2_ptrs = k_ptr + pid_bh * stride_k0 + pid_s * stride_k1 + (offs_d + HALF_D) * stride_k2

    # Pointers for cos/sin
    cos_ptrs = cos_ptr + pid_s * stride_cos0 + offs_d * stride_cos1
    sin_ptrs = sin_ptr + pid_s * stride_sin0 + offs_d * stride_sin1

    # Load data
    q1 = tl.load(q1_ptrs, mask=mask, other=0.0)
    q2 = tl.load(q2_ptrs, mask=mask, other=0.0)
    k1 = tl.load(k1_ptrs, mask=mask, other=0.0)
    k2 = tl.load(k2_ptrs, mask=mask, other=0.0)

    cos_vals = tl.load(cos_ptrs, mask=mask, other=0.0)
    sin_vals = tl.load(sin_ptrs, mask=mask, other=0.0)

    # RoPE rotation
    q_rot1 = q1 * cos_vals - q2 * sin_vals
    q_rot2 = q2 * cos_vals + q1 * sin_vals

    k_rot1 = k1 * cos_vals - k2 * sin_vals
    k_rot2 = k2 * cos_vals + k1 * sin_vals

    # Pointers for output q halves
    qo1_ptrs = q_out_ptr + pid_bh * stride_qo0 + pid_s * stride_qo1 + offs_d * stride_qo2
    qo2_ptrs = q_out_ptr + pid_bh * stride_qo0 + pid_s * stride_qo1 + (offs_d + HALF_D) * stride_qo2

    # Pointers for output k halves
    ko1_ptrs = k_out_ptr + pid_bh * stride_ko0 + pid_s * stride_ko1 + offs_d * stride_ko2
    ko2_ptrs = k_out_ptr + pid_bh * stride_ko0 + pid_s * stride_ko1 + (offs_d + HALF_D) * stride_ko2

    # Store results
    tl.store(qo1_ptrs, q_rot1, mask=mask)
    tl.store(qo2_ptrs, q_rot2, mask=mask)
    tl.store(ko1_ptrs, k_rot1, mask=mask)
    tl.store(ko2_ptrs, k_rot2, mask=mask)


def rope_triton(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
    """
    High-performance Triton implementation of RoPE.

    q, k: (B, H, S, D)
    cos, sin: (S, D//2)
    """
    assert q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda
    assert q.dtype == k.dtype == cos.dtype == sin.dtype
    assert q.ndim == 4 and k.ndim == 4
    B, H, S, D = q.shape
    assert D % 2 == 0
    HALF_D = D // 2

    # Flatten (B, H, S, D) -> (BH, S, D) for simpler indexing
    BH = B * H
    q_flat = q.contiguous().view(BH, S, D)
    k_flat = k.contiguous().view(BH, S, D)

    # Outputs
    q_out_flat = torch.empty_like(q_flat)
    k_out_flat = torch.empty_like(k_flat)

    # Truncate/broadcast cos/sin exactly like reference
    cos_trunc = cos[:S, :HALF_D].contiguous()
    sin_trunc = sin[:S, :HALF_D].contiguous()

    grid = lambda META: (
        BH,
        S,
        triton.cdiv(HALF_D, META["BLOCK_D"]),
    )

    rope_kernel[grid](
        q_flat, k_flat, cos_trunc, sin_trunc,
        q_out_flat, k_out_flat,
        BH, S, D, HALF_D,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        q_out_flat.stride(0), q_out_flat.stride(1), q_out_flat.stride(2),
        k_out_flat.stride(0), k_out_flat.stride(1), k_out_flat.stride(2),
        cos_trunc.stride(0), cos_trunc.stride(1),
        sin_trunc.stride(0), sin_trunc.stride(1),
        BLOCK_D=64,
    )

    # Restore original shape (B, H, S, D)
    q_rot = q_out_flat.view(B, H, S, D)
    k_rot = k_out_flat.view(B, H, S, D)

    return q_rot, k_rot, cos_trunc, sin_trunc


class ModelNew(nn.Module):
    """
    Triton-optimized RoPE module: replaces the PyTorch RoPE in Model.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, q, k, cos, sin):
        # Inputs are expected in shape (batch, n_heads, seq_len, head_dim)
        # and cos/sin in (seq_len, head_dim//2), matching the reference.
        q_rot, k_rot, cos_out, sin_out = rope_triton(
            q.to(device=cos.device),
            k.to(device=cos.device),
            cos,
            sin,
        )
        return q_rot, k_rot, cos_out, sin_out
