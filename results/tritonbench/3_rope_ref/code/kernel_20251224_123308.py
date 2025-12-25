import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Smaller, safer on register pressure / small head dims
        triton.Config({"BLOCK_D": 64}, num_warps=2, num_stages=2),
        # Good default for most head dims on 4090
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=2),
        # Aggressive for larger D, still safe for registers in this kernel
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=2),
    ],
    key=["HALF_D"],  # tune primarily on head dim (D/2)
)
@triton.jit
def rope_kernel_inplace(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    BH, S, D, HALF_D,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_cos0, stride_cos1,
    stride_sin0, stride_sin1,
    BLOCK_D: tl.constexpr,
):
    """
    In-place RoPE kernel.

    Layouts:
      q, k:    (BH, S, D)      strides (stride_q0, stride_q1, stride_q2)
      cos/sin: (S, HALF_D)     strides (stride_cos0, stride_cos1)

    Grid:
      pid_bh      in [0, BH)
      pid_s       in [0, S)
      pid_block_d in [0, ceil_div(HALF_D, BLOCK_D))

    Each program:
      - processes one (bh, s) pair
      - covers BLOCK_D elements along HALF_D dimension
      - rotates both q and k in-place
    """
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_block_d = tl.program_id(2)

    # Offsets along the HALF_D dimension
    offs_d = pid_block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < HALF_D

    # Help codegen for vectorized memory accesses
    tl.multiple_of(offs_d, BLOCK_D)

    # Base offsets for q/k for this (bh, s)
    q_base = pid_bh * stride_q0 + pid_s * stride_q1
    k_base = pid_bh * stride_k0 + pid_s * stride_k1

    # q/k pointers: first and second halves along D
    q1_ptrs = q_ptr + q_base + offs_d * stride_q2
    q2_ptrs = q1_ptrs + HALF_D * stride_q2

    k1_ptrs = k_ptr + k_base + offs_d * stride_k2
    k2_ptrs = k1_ptrs + HALF_D * stride_k2

    # cos/sin pointers: shared across BH, vary only with (s, d)
    cos_ptrs = cos_ptr + pid_s * stride_cos0 + offs_d * stride_cos1
    sin_ptrs = sin_ptr + pid_s * stride_sin0 + offs_d * stride_sin1

    # Load cos/sin once per (bh, s, d-block) and reuse for q and k
    cos_vals = tl.load(cos_ptrs, mask=mask, other=0.0)
    sin_vals = tl.load(sin_ptrs, mask=mask, other=0.0)

    # --- Rotate q in-place ---
    q1 = tl.load(q1_ptrs, mask=mask, other=0.0)
    q2 = tl.load(q2_ptrs, mask=mask, other=0.0)

    # [q1, q2] -> [q1*cos - q2*sin, q2*cos + q1*sin]
    q_rot1 = q1 * cos_vals - q2 * sin_vals
    q_rot2 = q2 * cos_vals + q1 * sin_vals

    tl.store(q1_ptrs, q_rot1, mask=mask)
    tl.store(q2_ptrs, q_rot2, mask=mask)

    # --- Rotate k in-place ---
    # Load k after writing q to reduce live registers
    k1 = tl.load(k1_ptrs, mask=mask, other=0.0)
    k2 = tl.load(k2_ptrs, mask=mask, other=0.0)

    # [k1, k2] -> [k1*cos - k2*sin, k2*cos + k1*sin]
    k_rot1 = k1 * cos_vals - k2 * sin_vals
    k_rot2 = k2 * cos_vals + k1 * sin_vals

    tl.store(k1_ptrs, k_rot1, mask=mask)
    tl.store(k2_ptrs, k_rot2, mask=mask)


def rope_triton(q: torch.Tensor, k: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
    """
    High-performance Triton implementation of RoPE.

    Inputs:
      q, k:  (B, H, S, D)
      cos, sin: (S, D//2)

    Returns:
      q_rot, k_rot: (B, H, S, D)
      cos_trunc, sin_trunc: (S, D//2)
    """
    assert q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda
    assert q.dtype == k.dtype == cos.dtype == sin.dtype
    assert q.ndim == 4 and k.ndim == 4

    B, H, S, D = q.shape
    assert D % 2 == 0
    HALF_D = D // 2

    # Flatten (B, H, S, D) -> (BH, S, D)
    BH = B * H
    assert BH > 0 and S > 0

    # Ensure contiguous so that view is valid and strides are simple
    q_flat = q.contiguous().view(BH, S, D)
    k_flat = k.contiguous().view(BH, S, D)

    # Match PyTorch reference truncation/broadcast semantics
    cos_trunc = cos[:S, :HALF_D].contiguous()
    sin_trunc = sin[:S, :HALF_D].contiguous()

    # Launch configuration:
    #   axis 0: BH
    #   axis 1: S
    #   axis 2: HALF_D / BLOCK_D
    def grid(meta):
        block_d = meta["BLOCK_D"]
        return (
            BH,
            S,
            triton.cdiv(HALF_D, block_d),
        )

    rope_kernel_inplace[grid](
        q_flat, k_flat, cos_trunc, sin_trunc,
        BH, S, D, HALF_D,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        cos_trunc.stride(0), cos_trunc.stride(1),
        sin_trunc.stride(0), sin_trunc.stride(1),
    )

    # q_flat / k_flat have been rotated in-place
    q_rot = q_flat.view(B, H, S, D)
    k_rot = k_flat.view(B, H, S, D)

    return q_rot, k_rot, cos_trunc, sin_trunc


class ModelNew(nn.Module):
    """
    Triton-optimized RoPE module.

    forward(q, k, cos, sin) -> (q_rotated, k_rotated, cos_out, sin_out)

    Shapes:
      q, k: (B, H, S, D)
      cos, sin: (S, D//2)
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, q, k, cos, sin):
        # Ensure q/k are on the same device as cos/sin
        dev = cos.device
        if q.device != dev:
            q = q.to(device=dev, non_blocking=True)
        if k.device != dev:
            k = k.to(device=dev, non_blocking=True)

        q_rot, k_rot, cos_out, sin_out = rope_triton(q, k, cos, sin)
        return q_rot, k_rot, cos_out, sin_out
