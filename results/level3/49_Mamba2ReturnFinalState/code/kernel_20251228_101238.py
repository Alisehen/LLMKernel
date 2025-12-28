import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

@triton.jit
def segsum_exp_kernel(
    x_ptr,  # [B, T]
    y_ptr,  # [B, T, T]
    B, T,
    stride_xb, stride_xt,
    stride_yb, stride_yi, stride_yj,
    BLOCK_B: tl.constexpr,  # number of batch elements per program
    BLOCK_T: tl.constexpr,  # max sequence length (power-of-2 >= T)
):
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B

    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < T

    # Load x for a block of batch elements: shape [BLOCK_B, BLOCK_T]
    x_ptrs = x_ptr + offs_b[:, None] * stride_xb + offs_t[None, :] * stride_xt
    x = tl.load(x_ptrs, mask=mask_b[:, None] & mask_t[None, :], other=0.0)

    # Mask for valid (b, j) positions within [0, T)
    mask_bt = mask_b[:, None] & mask_t[None, :]

    # For each row i, compute exp of segment sums sum_{k=i..j} x_k for all j >= i
    # Using cumsum on masked x.
    for i in range(0, BLOCK_T):
        # Guard against i >= T to avoid out-of-bounds writes
        if i < T:
            # mask for valid positions (b, j) with j >= i and j < T
            mask_row = mask_bt & (offs_t[None, :] >= i)

            # Zero out entries not contributing to this row's segment sums
            vals = tl.where(mask_row, x, 0.0)

            # Prefix sum along the time dimension
            prefix = tl.cumsum(vals, axis=1)

            # For j >= i: exp(sum_{k=i..j} x_k); for j < i: 0
            out = tl.where(mask_row, tl.exp(prefix), 0.0)

            y_ptrs = (
                y_ptr
                + offs_b[:, None] * stride_yb
                + i * stride_yi
                + offs_t[None, :] * stride_yj
            )
            # Store full row (including j < i as zeros), but only for j < T
            tl.store(y_ptrs, out, mask=mask_bt)


@triton.jit
def intra_states_kernel(
    B_ptr,        # [G, L, N]
    decay_ptr,    # [G, L]
    X_ptr,        # [G, L, P]
    states_ptr,   # [G, P, N]
    L, N, P,
    stride_bg, stride_bl, stride_bn,
    stride_dg, stride_dl,
    stride_xg, stride_xl, stride_xp,
    stride_sg, stride_sp, stride_sn,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    gid = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_p = offs_p < P
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    for l in range(0, L):
        decay = tl.load(decay_ptr + gid * stride_dg + l * stride_dl)

        b_ptrs = B_ptr + gid * stride_bg + l * stride_bl + offs_n * stride_bn
        B_l_n = tl.load(b_ptrs, mask=mask_n, other=0.0)

        x_ptrs = X_ptr + gid * stride_xg + l * stride_xl + offs_p * stride_xp
        X_l_p = tl.load(x_ptrs, mask=mask_p, other=0.0)

        Bl = B_l_n[None, :]      # [1, N]
        Xl = X_l_p[:, None]      # [P, 1]
        acc += (Xl * Bl) * decay

    out_ptrs = (
        states_ptr
        + gid * stride_sg
        + offs_p[:, None] * stride_sp
        + offs_n[None, :] * stride_sn
    )
    tl.store(out_ptrs, acc, mask=mask_p[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Python wrappers for Triton kernels
# ---------------------------------------------------------------------------

def segsum_exp_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Fused segsum + exp along the last dimension using Triton.

    Input:  x [..., T]
    Output: y [..., T, T] where
        y[..., i, j] = exp(sum_{k=i..j} x[..., k]) for j >= i
                       0 otherwise
    """
    assert x.is_cuda, "segsum_exp_triton requires CUDA tensors"
    orig_shape = x.shape
    T = orig_shape[-1]
    x_flat = x.contiguous().view(-1, T)
    B = x_flat.shape[0]

    y_flat = torch.empty(
        (B, T, T),
        device=x.device,
        dtype=x.dtype,
    )

    # Choose BLOCK_T as the smallest allowed power-of-two >= T
    if T <= 16:
        BLOCK_T = 16
    elif T <= 32:
        BLOCK_T = 32
    elif T <= 64:
        BLOCK_T = 64
    elif T <= 128:
        BLOCK_T = 128
    else:
        BLOCK_T = 256  # assumes T <= 256 in this workload

    grid = lambda META: (triton.cdiv(B, META["BLOCK_B"]),)

    segsum_exp_kernel[grid](
        x_flat,
        y_flat,
        B,
        T,
        x_flat.stride(0),
        x_flat.stride(1),
        y_flat.stride(0),
        y_flat.stride(1),
        y_flat.stride(2),
        BLOCK_B=32,
        BLOCK_T=BLOCK_T,
    )
    return y_flat.view(*orig_shape[:-1], T, T)


def intra_states_triton(
    B_blocks: torch.Tensor,      # [b, c, l, h, d_state]
    decay_states: torch.Tensor,  # [b, h, c, l]
    X_blocks: torch.Tensor,      # [b, c, l, h, d_head]
) -> torch.Tensor:
    """
    Compute:
        states = einsum("bclhn,bhcl,bclhp->bchpn",
                        B_blocks, decay_states, X_blocks)

    using Triton.

    Output: [b, c, h, d_head, d_state]
    """
    assert B_blocks.is_cuda, "intra_states_triton requires CUDA tensors"
    b, c, l, h, d_state = B_blocks.shape
    _, _, _, _, d_head = X_blocks.shape  # 5D shape unpack

    # Reorder to group (b, c, h) together
    B_perm = B_blocks.permute(0, 1, 3, 2, 4).contiguous()       # [b, c, h, l, n]
    X_perm = X_blocks.permute(0, 1, 3, 2, 4).contiguous()       # [b, c, h, l, p]
    decay_perm = decay_states.permute(0, 2, 1, 3).contiguous()  # [b, c, h, l]

    G = b * c * h
    B_flat = B_perm.view(G, l, d_state)      # [G, L, N]
    X_flat = X_perm.view(G, l, d_head)       # [G, L, P]
    decay_flat = decay_perm.view(G, l)       # [G, L]

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
        l,
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
        BLOCK_P=64,
        BLOCK_N=32,
    )

    states = states_flat.view(b, c, h, d_head, d_state)
    return states


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
        Forward pass implementing the SSD operation, optimized with Triton.

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
        X_blocks = X.view(b, c, l, h, d_head)  # [b, c, l, h, p]
        A_blocks_bc = self.A.view(b, c, l, h)  # [b, c, l, h]
        B_blocks = self.B.view(b, c, l, h, self.d_state)  # [b, c, l, h, n]

        # A_blocks: [b, h, c, l]
        A_blocks = A_blocks_bc.permute(0, 3, 1, 2).contiguous()
        A_cumsum = torch.cumsum(A_blocks, dim=-1)  # [b, h, c, l]

        # 2. Intra-chunk states
        decay_states = torch.exp(A_cumsum[:, :, :, -1:].clone() - A_cumsum)  # [b, h, c, l]
        states = intra_states_triton(B_blocks, decay_states, X_blocks)  # [b, c, h, p, n]

        # 3. Inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])  # [b, 1, h, p, n]

        states = torch.cat([initial_states, states], dim=1)  # [b, c+1, h, p, n]

        # last cumulative A per block: [b, h, c]
        A_last = A_cumsum[:, :, :, -1]  # [b, h, c]
        A_last_padded = F.pad(A_last, (1, 0))  # [b, h, c+1]

        decay_chunk = segsum_exp_triton(A_last_padded)  # [b, h, c+1, c+1]

        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        return new_states[:, -1]
