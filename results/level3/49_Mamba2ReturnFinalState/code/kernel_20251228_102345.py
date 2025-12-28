# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def decay_from_A_kernel(
    A_ptr,         # [G, L]
    decay_ptr,     # [G, L]
    A_last_ptr,    # [G]
    G,
    stride_ag, stride_al,
    stride_dg, stride_dl,
    stride_lg,
    L: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    For each row g in [0, G):

        Let a[g, :] be length-L.
        s[i] = sum_{t=0..i} a[t]
        S_total = s[L-1]
        decay[g, i] = exp(S_total - s[i])  for i in [0, L)

    Also outputs A_last[g] = S_total.

    Grid: 1D over rows (g). Each program processes full length-L row.
    """
    pid = tl.program_id(0)
    offs_l = tl.arange(0, BLOCK_L)

    mask_g = pid < G
    mask_l = offs_l < L
    mask = mask_g & mask_l

    # Load row A[g, :]
    A_row_ptr = A_ptr + pid * stride_ag + offs_l * stride_al
    a = tl.load(A_row_ptr, mask=mask, other=0.0)
    a = a.to(tl.float32)

    # Prefix sum and total
    s = tl.cumsum(a, axis=0)  # [BLOCK_L]
    S_total = s[L - 1]        # scalar

    # decay = exp(S_total - s)
    decay = tl.exp(S_total - s)

    # Store decay[g, :]
    decay_row_ptr = decay_ptr + pid * stride_dg + offs_l * stride_dl
    tl.store(decay_row_ptr, decay, mask=mask)

    # Store A_last[g] = S_total
    A_last_row_ptr = A_last_ptr + pid * stride_lg
    tl.store(A_last_row_ptr, S_total, mask=mask_g)


@triton.jit
def intra_states_kernel(
    B_ptr,        # [G, L, N]
    decay_ptr,    # [G, L]
    X_ptr,        # [G, L, P]
    states_ptr,   # [G, P, N]
    G, N, P,
    stride_bg, stride_bl, stride_bn,
    stride_dg, stride_dl,
    stride_xg, stride_xl, stride_xp,
    stride_sg, stride_sp, stride_sn,
    L: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute, for each group g in [0, G):

        states[g, p, n] = sum_{l=0..L-1} decay[g, l] * X[g, l, p] * B[g, l, n]

    where:
        B_ptr:     [G, L, N]
        decay_ptr: [G, L]
        X_ptr:     [G, L, P]
        states:    [G, P, N]

    Grid: (G, ceil(P/BLOCK_P), ceil(N/BLOCK_N)).
    Each program computes a (BLOCK_P x BLOCK_N) tile of [P, N] for fixed g.
    """
    gid = tl.program_id(0)  # group index
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Output tile coordinates
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_g = gid < G
    mask_p = offs_p < P
    mask_n = offs_n < N
    mask_out = mask_g & mask_p[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    # Base pointers for this g
    B_base_g = B_ptr + gid * stride_bg
    X_base_g = X_ptr + gid * stride_xg
    D_base_g = decay_ptr + gid * stride_dg

    # Loop over sequence dimension L (compile-time constant)
    for l in range(0, L):
        # decay[g, l] as scalar
        decay = tl.load(D_base_g + l * stride_dl, mask=mask_g, other=0.0)
        decay = decay.to(tl.float32)

        # B[g, l, n] -> [BLOCK_N]
        b_ptrs = B_base_g + l * stride_bl + offs_n * stride_bn
        B_l_n = tl.load(b_ptrs, mask=mask_g & mask_n, other=0.0)
        B_l_n = B_l_n.to(tl.float32)

        # X[g, l, p] -> [BLOCK_P]
        x_ptrs = X_base_g + l * stride_xl + offs_p * stride_xp
        X_l_p = tl.load(x_ptrs, mask=mask_g & mask_p, other=0.0)
        X_l_p = X_l_p.to(tl.float32)

        # Outer product X[g,l,p] * B[g,l,n] for tile (p, n)
        Bl = B_l_n[None, :]      # [1, BLOCK_N]
        Xl = X_l_p[:, None]      # [BLOCK_P, 1]

        acc += (Xl * Bl) * decay

    # Store result states[g, p, n]
    out_ptrs = (
        states_ptr
        + gid * stride_sg
        + offs_p[:, None] * stride_sp
        + offs_n[None, :] * stride_sn
    )
    tl.store(out_ptrs, acc, mask=mask_out)


@triton.jit
def inter_state_recurrence_kernel(
    states_ptr,    # [G, C, P, N]
    A_last_ptr,    # [G, C]
    init_ptr,      # [G, P, N]
    out_ptr,       # [G, P, N]
    G, P, N,
    stride_sg, stride_sc, stride_sp, stride_sn,
    stride_ag, stride_ac,
    stride_ig, stride_ip, stride_in,
    stride_og, stride_op, stride_on,
    C: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    For each group g and (p, n), compute the inter-chunk recurrence:

        s_0 = init[g, p, n]
        for k in 0..C-1:
            s_{k+1} = exp(A_last[g, k]) * s_k + states[g, k, p, n]

        out[g, p, n] = s_C

    where:
        states: [G, C, P, N]
        A_last: [G, C]
        init:   [G, P, N]
        out:    [G, P, N]

    Grid: (G, ceil(P/BLOCK_P), ceil(N/BLOCK_N)).
    """
    pid_g = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_g = pid_g < G
    mask_p = offs_p < P
    mask_n = offs_n < N
    mask = mask_g & mask_p[:, None] & mask_n[None, :]

    # Load initial state s_0[g, p, n]
    init_ptrs = (
        init_ptr
        + pid_g * stride_ig
        + offs_p[:, None] * stride_ip
        + offs_n[None, :] * stride_in
    )
    s = tl.load(init_ptrs, mask=mask, other=0.0)
    s = s.to(tl.float32)

    # Base pointers for this g
    states_base_g = states_ptr + pid_g * stride_sg
    A_last_base_g = A_last_ptr + pid_g * stride_ag

    # Recurrence over chunks k
    for k in range(0, C):
        # a = A_last[g, k]
        a_ptr = A_last_base_g + k * stride_ac
        a = tl.load(a_ptr, mask=mask_g, other=0.0)
        a = a.to(tl.float32)
        a_exp = tl.exp(a)

        # states[g, k, p, n]
        states_ptrs = (
            states_base_g
            + k * stride_sc
            + offs_p[:, None] * stride_sp
            + offs_n[None, :] * stride_sn
        )
        contrib = tl.load(states_ptrs, mask=mask, other=0.0)
        contrib = contrib.to(tl.float32)

        s = a_exp * s + contrib

    # Store final state
    out_ptrs = (
        out_ptr
        + pid_g * stride_og
        + offs_p[:, None] * stride_op
        + offs_n[None, :] * stride_on
    )
    tl.store(out_ptrs, s, mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers for Triton kernels
# ---------------------------------------------------------------------------

def compute_decay_and_last_triton(A_blocks: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    A_blocks: [b, h, c, l]
    Returns:
        decay_states: [b, h, c, l]
        A_last:       [b, h, c]
    """
    assert A_blocks.is_cuda, "compute_decay_and_last_triton requires CUDA tensors"
    b, h, c, l = A_blocks.shape

    A_flat = A_blocks.contiguous().view(-1, l)  # [G_dec, L]
    G_dec = A_flat.shape[0]

    decay_flat = torch.empty_like(A_flat)
    A_last_flat = torch.empty(
        G_dec,
        device=A_blocks.device,
        dtype=A_blocks.dtype,
    )

    # Choose BLOCK_L as the smallest allowed power-of-two >= l
    if l <= 16:
        BLOCK_L = 16
    elif l <= 32:
        BLOCK_L = 32
    elif l <= 64:
        BLOCK_L = 64
    else:
        BLOCK_L = 128  # assumes block_len <= 128

    grid = lambda META: (G_dec,)

    decay_from_A_kernel[grid](
        A_flat,
        decay_flat,
        A_last_flat,
        G_dec,
        A_flat.stride(0),
        A_flat.stride(1),
        decay_flat.stride(0),
        decay_flat.stride(1),
        A_last_flat.stride(0),
        L=l,
        BLOCK_L=BLOCK_L,
        num_warps=2,
    )

    decay_blocks = decay_flat.view(b, h, c, l)
    A_last = A_last_flat.view(b, h, c)
    return decay_blocks, A_last


def intra_states_triton(
    B_blocks: torch.Tensor,      # [b, c, l, h, d_state]
    decay_states: torch.Tensor,  # [b, h, c, l]
    X_blocks: torch.Tensor,      # [b, c, l, h, d_head]
) -> torch.Tensor:
    """
    Compute:
        states = einsum("bclhn,bhcl,bclhp->bchpn",
                        B_blocks, decay_states, X_blocks)

    Output: [b, c, h, d_head, d_state]
    """
    assert B_blocks.is_cuda, "intra_states_triton requires CUDA tensors"

    b, c, l, h, d_state = B_blocks.shape
    _, _, _, _, d_head = X_blocks.shape

    # Reorder to group (b, c, h) together
    B_perm = B_blocks.permute(0, 1, 3, 2, 4).contiguous()       # [b, c, h, l, n]
    X_perm = X_blocks.permute(0, 1, 3, 2, 4).contiguous()       # [b, c, h, l, p]
    decay_perm = decay_states.permute(0, 2, 1, 3).contiguous()  # [b, c, h, l]

    G = b * c * h
    B_flat = B_perm.view(G, l, d_state)    # [G, L, N]
    X_flat = X_perm.view(G, l, d_head)     # [G, L, P]
    decay_flat = decay_perm.view(G, l)     # [G, L]

    states_flat = torch.empty(
        (G, d_head, d_state),
        device=B_blocks.device,
        dtype=B_blocks.dtype,
    )

    grid = lambda META: (
        G,
        triton.cdiv(d_head, META["BLOCK_P"]),
        triton.cdiv(d_state, META["BLOCK_N"]),
    )

    intra_states_kernel[grid](
        B_flat,
        decay_flat,
        X_flat,
        states_flat,
        G,
        d_state,
        d_head,
        B_flat.stride(0),
        B_flat.stride(1),
        B_flat.stride(2),
        decay_flat.stride(0),
        decay_flat.stride(1),
        X_flat.stride(0),
        X_flat.stride(1),
        X_flat.stride(2),
        states_flat.stride(0),
        states_flat.stride(1),
        states_flat.stride(2),
        L=l,
        BLOCK_P=64,
        BLOCK_N=32,
        num_warps=4,
    )

    states = states_flat.view(b, c, h, d_head, d_state)
    return states


def inter_state_recurrence_triton(
    states: torch.Tensor,   # [b, c, h, d_head, d_state]
    A_last: torch.Tensor,   # [b, h, c]
    init_states: torch.Tensor,  # [b, h, d_head, d_state]
) -> torch.Tensor:
    """
    Compute inter-chunk recurrence over c:

        s_{k+1} = exp(A_last[..., k]) * s_k + states[:, k]

    Inputs:
        states:     [b, c, h, p, n]
        A_last:     [b, h, c]
        init_states:[b, h, p, n]

    Output:
        final_states: [b, h, p, n]
    """
    assert states.is_cuda, "inter_state_recurrence_triton requires CUDA tensors"

    b, c, h, p, n = states.shape
    assert A_last.shape == (b, h, c)
    assert init_states.shape == (b, h, p, n)

    # Reorder to group (b, h) together
    states_perm = states.permute(0, 2, 1, 3, 4).contiguous()  # [b, h, c, p, n]
    G = b * h
    states_flat = states_perm.view(G, c, p, n)               # [G, C, P, N]

    A_last_flat = A_last.contiguous().view(G, c)             # [G, C]
    init_flat = init_states.contiguous().view(G, p, n)       # [G, P, N]
    out_flat = torch.empty_like(init_flat)                   # [G, P, N]

    grid = lambda META: (
        G,
        triton.cdiv(p, META["BLOCK_P"]),
        triton.cdiv(n, META["BLOCK_N"]),
    )

    inter_state_recurrence_kernel[grid](
        states_flat,
        A_last_flat,
        init_flat,
        out_flat,
        G,
        p,
        n,
        states_flat.stride(0),
        states_flat.stride(1),
        states_flat.stride(2),
        states_flat.stride(3),
        A_last_flat.stride(0),
        A_last_flat.stride(1),
        init_flat.stride(0),
        init_flat.stride(1),
        init_flat.stride(2),
        out_flat.stride(0),
        out_flat.stride(1),
        out_flat.stride(2),
        C=c,
        BLOCK_P=64,
        BLOCK_N=32,
        num_warps=4,
    )

    final_states = out_flat.view(b, h, p, n)
    return final_states


# ---------------------------------------------------------------------------
# Optimized Model
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(ModelNew, self).__init__()

        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation with Triton-optimized
        intra-chunk and inter-chunk computations.

        X: (batch, length, n_heads, d_head)
        Returns: final state (batch, n_heads, d_head, d_state)
        """
        b, T, h, d_head = X.shape
        assert b == self.batch_size
        assert T == self.seq_length
        assert h == self.n_heads
        assert d_head == self.d_head

        c = T // self.block_len
        l = self.block_len

        # Reshape into blocks
        X_blocks = X.view(b, c, l, h, d_head).contiguous()               # [b, c, l, h, p]
        B_blocks = self.B.view(b, c, l, h, self.d_state).contiguous()    # [b, c, l, h, n]

        # Prepare A into [b, h, c, l] and compute decay + A_last via Triton
        A_blocks_bc = self.A.view(b, c, l, h)                            # [b, c, l, h]
        A_blocks = A_blocks_bc.permute(0, 3, 1, 2).contiguous()          # [b, h, c, l]
        decay_states, A_last = compute_decay_and_last_triton(A_blocks)   # [b, h, c, l], [b, h, c]

        # Intra-chunk states (Triton kernel)
        states = intra_states_triton(B_blocks, decay_states, X_blocks)   # [b, c, h, p, n]

        # Inter-chunk recurrence (Triton kernel)
        if initial_states is None:
            init_states = torch.zeros(
                (b, h, d_head, self.d_state),
                device=X.device,
                dtype=X.dtype,
            )
        else:
            assert initial_states.shape == (b, h, d_head, self.d_state)
            init_states = initial_states

        current_state = inter_state_recurrence_triton(states, A_last, init_states)  # [b, h, p, n]

        return current_state
