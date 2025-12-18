import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 16}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 64}, num_warps=4, num_stages=2),
    ],
    key=['T'],
)
@triton.jit
def segsum_from_cumsum_kernel(
    cs_ptr, out_ptr,
    T,
    stride_cs_b, stride_cs_t,
    stride_out_b, stride_out_i, stride_out_j,
    BLOCK: tl.constexpr,
):
    """
    Compute segmented sums from a precomputed cumulative sum tensor.

    Given cs[b, t] = cumsum(x[b, t]) along t, produce out[b, i, j] = cs[b, i] - cs[b, j]
    for i >= j (lower-triangular including diagonal). Positions with i < j or out-of-bounds
    are left untouched (assumed pre-initialized to -inf by the caller).

    Shapes:
      cs:  (B, T)
      out: (B, T, T)
    """
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)

    # Skip tiles that are strictly above the diagonal: (pid_i+1)*BLOCK <= pid_j*BLOCK
    # => max_i < min_j => no element with i >= j in this tile.
    if (pid_i + 1) <= pid_j:
        return

    # Column indices for this tile
    offs_j = pid_j * BLOCK + tl.arange(0, BLOCK)
    mask_j = offs_j < T

    # Base pointers for this batch row
    cs_row_ptr = cs_ptr + pid_b * stride_cs_b
    out_row_ptr = out_ptr + pid_b * stride_out_b

    # Load cs[b, offs_j] once per tile
    cs_j = tl.load(cs_row_ptr + offs_j * stride_cs_t, mask=mask_j, other=0.0)

    # Loop over rows in this tile; BLOCK is constexpr so this unrolls
    for ii in range(BLOCK):
        i_idx = pid_i * BLOCK + ii
        mask_i = i_idx < T

        # Combined in-bounds mask
        # We only store to positions that are within [0, T)
        # and in the lower triangle (i_idx >= offs_j).
        tri_mask = (i_idx >= offs_j) & mask_j & mask_i

        # If no element in this row is in-bounds & lower-triangular, skip work.
        # Triton does not support short-circuit on vector masks, so we rely on mask in store.
        cs_i = tl.load(cs_row_ptr + i_idx * stride_cs_t, mask=mask_i, other=0.0)

        # Broadcast subtraction: cs[b, i] - cs[b, j]
        diff_row = cs_i - cs_j

        # Compute output addresses for this row
        row_out_ptrs = out_row_ptr + i_idx * stride_out_i + offs_j * stride_out_j

        # Store only where i >= j and indices are valid; other positions keep prefilled -inf
        tl.store(row_out_ptrs, diff_row, mask=tri_mask)


def segsum_from_cumsum_triton(cs: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of the segsum operation given a precomputed cumsum:

      Given cs = torch.cumsum(x, dim=-1),
      segsum(cs) = cs[..., :, None] - cs[..., None, :]
      masked to lower triangle (i >= j) with -inf elsewhere.

    Works for arbitrary leading dimensions (..., T).
    """
    assert cs.is_cuda, "Input must be on CUDA device for Triton kernel"

    orig_shape = cs.shape  # (..., T)
    T = orig_shape[-1]
    if T <= 0:
        raise ValueError("Last dimension T must be > 0")

    # Flatten all leading dimensions into a single batch dimension B
    cs_reshaped = cs.reshape(-1, T).contiguous()  # (B, T)
    B = cs_reshaped.shape[0]

    # Allocate and pre-fill output (B, T, T) with -inf.
    # Kernel only writes lower-triangular valid entries.
    out = torch.full(
        (B, T, T),
        -float("inf"),
        device=cs.device,
        dtype=cs.dtype,
    )

    def grid(meta):
        block = meta['BLOCK']
        return (
            B,
            triton.cdiv(T, block),
            triton.cdiv(T, block),
        )

    segsum_from_cumsum_kernel[grid](
        cs_reshaped, out,
        T,
        cs_reshaped.stride(0), cs_reshaped.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
    )

    # Reshape back to original shape with extra T dimension
    return out.reshape(*orig_shape, T)


def segsum_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Generic segsum wrapper that computes cumsum using PyTorch and then
    calls the Triton kernel on the cumsum.
    """
    assert x.is_cuda, "Input must be on CUDA device for Triton kernel"

    cs = torch.cumsum(x, dim=-1)
    return segsum_from_cumsum_triton(cs)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation with Triton-optimized segsum.
        """
        super(ModelNew, self).__init__()

        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation, using Triton for the segsum
        computations. Segsum over A_blocks reuses the already-computed cumsum
        to avoid redundant work.
        """
        from einops import rearrange
        import torch.nn.functional as F

        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]

        # A_blocks: (b, c, l, h) -> (b, h, c, l)
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")

        # Cumulative sum within each block along the last dimension
        A_cumsum = torch.cumsum(A_blocks, dim=-1)

        # 1. Compute diagonal block outputs
        #    Use precomputed A_cumsum to avoid an extra cumsum inside segsum.
        L = torch.exp(segsum_from_cumsum_triton(A_cumsum))
        Y_diag = torch.einsum(
            "bclhn,bcshn,bhcls,bcshp->bclhp",
            C_blocks,
            B_blocks,
            L,
            X_blocks,
        )

        # 2. Compute intra-chunk states
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum(
            "bclhn,bhcl,bclhp->bchpn",
            B_blocks,
            decay_states,
            X_blocks,
        )

        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        # Chunk-level decays: segsum over padded per-chunk sums
        decay_chunk = torch.exp(
            segsum_triton(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
        )
        new_states = torch.einsum(
            "bhzc,bchpn->bzhpn",
            decay_chunk,
            states,
        )
        return new_states[:, -1]
