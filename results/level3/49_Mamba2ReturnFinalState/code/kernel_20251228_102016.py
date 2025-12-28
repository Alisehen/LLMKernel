import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def y_diag_kernel(
    B_ptr, C_ptr, X_ptr, u_ptr, v_ptr, Y_ptr,
    M, P,
    stride_bm, stride_bl, stride_bn,
    stride_cm, stride_cl, stride_cn,
    stride_xm, stride_xl, stride_xp,
    stride_um, stride_ul,
    stride_vm, stride_vl,
    stride_ym, stride_yl, stride_yp,
    BLOCK_P: tl.constexpr, D_STATE: tl.constexpr, BLOCK_L: tl.constexpr,
):
    """
    Optimized linear-time scan kernel for Y_diag.

    Input (flattened over (b, c, h) -> M):
      B_flat: [M, BLOCK_L, D_STATE]
      C_flat: [M, BLOCK_L, D_STATE]
      X_flat: [M, BLOCK_L, P]
      u_flat: [M, BLOCK_L]   where u_t = exp(cumsum(A)_t)
      v_flat: [M, BLOCK_L]   where v_s = exp(-cumsum(A)_s)
      Y_flat: [M, BLOCK_L, P]

    For each group m in [0, M), time t in [0, BLOCK_L), and feature p:
      L[t,s] = u[t] * v[s] for s <= t else 0
      Y[t,p] = sum_{s <= t, n} C[t,n] * B[s,n] * L[t,s] * X[s,p]

    We implement this in O(L * D_STATE * P) using the prefix state:
      S_t[n,p] = sum_{s <= t} v[s] * B[s,n] * X[s,p]
      Y[t,p]   = u[t] * sum_n C[t,n] * S_t[n,p]
    """
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)

    # Offsets along feature dimension
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = offs_p < P

    # Base pointers for this (b, c, h) group
    B_row = B_ptr + pid_m * stride_bm
    C_row = C_ptr + pid_m * stride_cm
    X_row = X_ptr + pid_m * stride_xm
    u_row = u_ptr + pid_m * stride_um
    v_row = v_ptr + pid_m * stride_vm
    Y_row = Y_ptr + pid_m * stride_ym

    # Offsets for state dimension
    offs_n = tl.arange(0, D_STATE)

    # Persistent prefix state S_t[p, n] in registers (fp32 for accuracy):
    # S[p, n] = sum_{s <= t} v[s] * B[s,n] * X[s,p]
    S = tl.zeros((BLOCK_P, D_STATE), dtype=tl.float32)

    # Time loop inside kernel (linear in sequence length within block)
    for t in range(0, BLOCK_L):
        # Load v[t] (scalar)
        v_t = tl.load(v_row + t * stride_vl).to(tl.float32)

        # Load X[m, t, p] for this tile
        x_vec = tl.load(
            X_row + t * stride_xl + offs_p * stride_xp,
            mask=p_mask,
            other=0.0,
        ).to(tl.float32)

        # Load B[m, t, n] as a vector
        B_vec = tl.load(
            B_row + t * stride_bl + offs_n * stride_bn
        ).to(tl.float32)

        # Update prefix state:
        # S[p, n] += v_t * B_vec[n] * x_vec[p]
        coef = v_t * B_vec  # [D_STATE]
        S += x_vec[:, None] * coef[None, :]

        # Load u[t] (scalar)
        u_t = tl.load(u_row + t * stride_ul).to(tl.float32)

        # Load C[m, t, n] as a vector
        C_vec = tl.load(
            C_row + t * stride_cl + offs_n * stride_cn
        ).to(tl.float32)

        # Compute Y[t, p] = u_t * sum_n C[t,n] * S[p,n]
        # First: dot over n -> [BLOCK_P]
        y_vec = tl.sum(S * C_vec[None, :], axis=1)
        y_vec = y_vec * u_t

        # Store result
        tl.store(
            Y_row + t * stride_yl + offs_p * stride_yp,
            y_vec,
            mask=p_mask,
        )


def fused_y_diag(C_blocks, B_blocks, X_blocks, A_cumsum):
    """
    Fused computation of Y_diag using Triton with an O(L * d_state * d_head)
    online scan instead of the original O(L^2) formulation.

    Original PyTorch:
      L = torch.exp(segsum(A_blocks))   # A_blocks: (b, h, c, l)
      Y_diag = torch.einsum(
          "bclhn,bcshn,bhcls,bcshp->bclhp",
          C_blocks, B_blocks, L, X_blocks
      )

    Using L[t,s] = exp(segsum(A)[t,s]) = u[t] * v[s] (for t >= s):
      u = exp(cumsum(A)), v = exp(-cumsum(A))

    We compute:
      S_t[n,p] = sum_{s <= t} v[s] * B[s,n] * X[s,p]
      Y[t,p]   = u[t] * sum_n C[t,n] * S_t[n,p]
    """
    # Shapes
    b, c, l, h, n_state = B_blocks.shape
    _, _, _, _, p = X_blocks.shape

    # A_cumsum: (b, h, c, l)
    # Precompute u_t = exp(cumsum(A)_t), v_s = exp(-cumsum(A)_s)
    u = torch.exp(A_cumsum)      # (b, h, c, l)
    v = torch.exp(-A_cumsum)     # (b, h, c, l)

    # Reorder u, v to (b, c, h, l) so we can flatten (b, c, h)
    u = u.permute(0, 2, 1, 3).contiguous()  # (b, c, h, l)
    v = v.permute(0, 2, 1, 3).contiguous()  # (b, c, h, l)

    # Reorder B, C, X to (b, c, h, l, *)
    B_perm = B_blocks.permute(0, 1, 3, 2, 4).contiguous()  # (b, c, h, l, n_state)
    C_perm = C_blocks.permute(0, 1, 3, 2, 4).contiguous()  # (b, c, h, l, n_state)
    X_perm = X_blocks.permute(0, 1, 3, 2, 4).contiguous()  # (b, c, h, l, p)

    # Flatten outer dimensions (b, c, h) -> M
    M = b * c * h
    B_flat = B_perm.view(M, l, n_state)
    C_flat = C_perm.view(M, l, n_state)
    X_flat = X_perm.view(M, l, p)
    u_flat = u.view(M, l)
    v_flat = v.view(M, l)

    # Allocate output
    Y_flat = torch.empty((M, l, p), device=X_blocks.device, dtype=X_blocks.dtype)

    # Strides
    stride_bm, stride_bl, stride_bn = B_flat.stride()
    stride_cm, stride_cl, stride_cn = C_flat.stride()
    stride_xm, stride_xl, stride_xp = X_flat.stride()
    stride_um, stride_ul = u_flat.stride()
    stride_vm, stride_vl = v_flat.stride()
    stride_ym, stride_yl, stride_yp = Y_flat.stride()

    # Tile sizes
    BLOCK_L = l              # full block length processed per program in time
    BLOCK_P = 64             # feature tile; power-of-two as required

    # Grid: (groups over (b,c,h), feature tiles)
    grid = lambda META: (
        M,
        triton.cdiv(p, META['BLOCK_P']),
    )

    y_diag_kernel[grid](
        B_flat, C_flat, X_flat, u_flat, v_flat, Y_flat,
        M, p,
        stride_bm, stride_bl, stride_bn,
        stride_cm, stride_cl, stride_cn,
        stride_xm, stride_xl, stride_xp,
        stride_um, stride_ul,
        stride_vm, stride_vl,
        stride_ym, stride_yl, stride_yp,
        BLOCK_P=BLOCK_P, D_STATE=n_state, BLOCK_L=BLOCK_L,
    )

    # Reshape back to (b, c, l, h, p)
    Y_perm = Y_flat.view(b, c, h, l, p)
    Y_diag = Y_perm.permute(0, 1, 3, 2, 4).contiguous()  # (b, c, l, h, p)
    return Y_diag


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Triton-optimized Mamba Structured State Space model.
        """
        super(ModelNew, self).__init__()

        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        # Parameters: same shapes as original model
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def segsum(self, x):
        """Segment sum along the last dimension (naive implementation, as in original)."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation, with the heavy
        diagonal-block computation accelerated by Triton.
        """
        b, seq_len, h, p = X.shape
        l = self.block_len
        c = seq_len // l

        # 0. Rearrange into blocks/chunks: "b (c l) ... -> b c l ..."
        X_blocks = X.view(b, c, l, h, p)
        A_blocks = self.A.view(b, c, l, h)
        B_blocks = self.B.view(b, c, l, h, self.d_state)
        C_blocks = self.C.view(b, c, l, h, self.d_state)

        # A_blocks: (b, c, l, h) -> (b, h, c, l)
        A_bhcl = A_blocks.permute(0, 3, 1, 2).contiguous()
        A_cumsum = torch.cumsum(A_bhcl, dim=-1)  # (b, h, c, l)

        # 1. Compute diagonal block outputs (Y_diag) with fused Triton kernel
        Y_diag = fused_y_diag(C_blocks, B_blocks, X_blocks, A_cumsum)

        # 2. Compute intra-chunk states (same as original, using A_cumsum)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # (b, h, c, l)
        states = torch.einsum(
            "bclhn,bhcl,bclhp->bchpn",
            B_blocks, decay_states, X_blocks
        )  # (b, c, h, p, d_state)

        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])  # (b, 1, h, p, d_state)

        states = torch.cat([initial_states, states], dim=1)  # (b, c+1, h, p, d_state)

        # Prepare chunk-level A sums for recurrence
        A_end = A_cumsum[:, :, :, -1]  # (b, h, c)
        zero_pad = torch.zeros_like(A_end[..., :1])  # (b, h, 1)
        A_padded = torch.cat([zero_pad, A_end], dim=-1)  # (b, h, c+1)

        decay_chunk = torch.exp(self.segsum(A_padded))  # (b, h, c+1, c+1)
        new_states = torch.einsum(
            "bhzc,bchpn->bzhpn",
            decay_chunk, states
        )  # (b, c+1, h, p, d_state)

        # Return final state as in original
        return new_states[:, -1]
