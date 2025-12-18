import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Conservative baseline
        triton.Config({"BLOCK": 32}, num_warps=4, num_stages=2),
        # More aggressive config for compute-heavy tiles
        triton.Config({"BLOCK": 64}, num_warps=8, num_stages=3),
    ],
    key=["T"],
)
@triton.jit
def segsum_exp_kernel(
    x_cumsum_ptr,  # [M, T]
    y_ptr,         # [M, T, T]
    M, T,
    stride_xm, stride_xt,
    stride_ym, stride_yi, stride_yj,
    BLOCK: tl.constexpr,
):
    """
    Compute y = exp(segsum(x_cumsum)) where:
      segsum(x_cumsum)[m, i, j] = x_cumsum[m, i] - x_cumsum[m, j] for i >= j
                               = 0                            otherwise
    Only lower-triangular tiles are computed; upper-triangular entries are
    assumed to be pre-zeroed in y.
    """
    pid_m = tl.program_id(0)
    pid_t = tl.program_id(1)

    if pid_m >= M:
        return

    # Map linear triangular tile id -> (bi, bj) with 0 <= bj <= bi
    pid_t_i32 = tl.cast(pid_t, tl.int32)
    pid_t_f32 = tl.cast(pid_t_i32, tl.float32)

    # bi = floor( (sqrt(1 + 8*pid_t) - 1) / 2 )
    sqrt_arg = 1.0 + 8.0 * pid_t_f32
    bi_f = (tl.sqrt(sqrt_arg) - 1.0) * 0.5
    bi = tl.cast(tl.floor(bi_f), tl.int32)

    # First triangular index of row bi: bi * (bi + 1) // 2
    row_start = bi * (bi + 1) // 2
    bj = pid_t_i32 - row_start

    # Offsets in i/j dimensions
    offs_i = bi * BLOCK + tl.arange(0, BLOCK)
    offs_j = bj * BLOCK + tl.arange(0, BLOCK)

    mask_i = offs_i < T
    mask_j = offs_j < T

    # Base pointers for row m
    base_x = x_cumsum_ptr + pid_m * stride_xm
    base_y = y_ptr + pid_m * stride_ym

    # Load cumsum values for i and j indices
    x_i_ptrs = base_x + offs_i * stride_xt
    x_j_ptrs = base_x + offs_j * stride_xt

    cumsum_i = tl.load(x_i_ptrs, mask=mask_i, other=0.0)  # [BLOCK]
    cumsum_j = tl.load(x_j_ptrs, mask=mask_j, other=0.0)  # [BLOCK]

    # Broadcast to [BLOCK, BLOCK] and compute exponentials
    diff = cumsum_i[:, None] - cumsum_j[None, :]
    out = tl.exp(diff)

    valid_mask = mask_i[:, None] & mask_j[None, :]

    # For tiles strictly below the diagonal (bi > bj), all (i, j) in the tile
    # satisfy i >= j, so no triangular masking is needed.
    # For diagonal tiles (bi == bj), apply lower-triangular mask.
    same_block = bi == bj
    if same_block:
        tri_mask = offs_i[:, None] >= offs_j[None, :]
        store_mask = valid_mask & tri_mask
    else:
        store_mask = valid_mask

    # Compute output pointers and store only for lower-triangular positions
    y_ptrs = base_y + offs_i[:, None] * stride_yi + offs_j[None, :] * stride_yj
    tl.store(y_ptrs, out, mask=store_mask)


def segsum_exp_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Compute exp(segsum(x)) along the last dimension using Triton.

    segsum(x)[..., i, j] = cumsum(x)[..., i] - cumsum(x)[..., j] for i >= j,
                           = -inf otherwise
    exp(segsum(x)) therefore is:
        exp(cumsum[i] - cumsum[j]) for i >= j
        0                         otherwise

    Args:
        x: tensor of shape (..., T), CUDA tensor.

    Returns:
        Tensor of shape (..., T, T) with exp(segsum(x)).
    """
    assert x.is_cuda, "segsum_exp_triton expects a CUDA tensor"
    x_cumsum = torch.cumsum(x, dim=-1).contiguous()
    *prefix_shape, T = x_cumsum.shape
    if T == 0:
        return x_cumsum.new_empty(*prefix_shape, 0, 0)

    M = x_cumsum.numel() // T
    x_2d = x_cumsum.view(M, T)

    # Output tensor: pre-zero so we can skip upper-triangular tiles
    y = torch.zeros((M, T, T), device=x.device, dtype=x.dtype)

    stride_xm, stride_xt = x_2d.stride()
    stride_ym, stride_yi, stride_yj = y.stride()

    def grid(meta):
        BLOCK = meta["BLOCK"]
        num_blocks = (T + BLOCK - 1) // BLOCK
        # Number of tiles in lower-triangular (including diagonal)
        num_tri_tiles = num_blocks * (num_blocks + 1) // 2
        return (M, num_tri_tiles)

    segsum_exp_kernel[grid](
        x_2d,
        y,
        M,
        T,
        stride_xm,
        stride_xt,
        stride_ym,
        stride_yi,
        stride_yj,
    )

    return y.view(*prefix_shape, T, T)


def segsum_exp_naive(x: torch.Tensor) -> torch.Tensor:
    """
    Fallback implementation of exp(segsum(x)) using PyTorch.
    """
    T = x.size(-1)
    if T == 0:
        return x.new_empty(*x.shape, 0)

    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]

    # Lower-triangular mask (including diagonal)
    mask = torch.tril(
        torch.ones(T, T, device=x.device, dtype=torch.bool),
        diagonal=0,
    )
    # Broadcast mask to leading dimensions
    while mask.dim() < x_segsum.dim():
        mask = mask.unsqueeze(0)
    x_segsum = x_segsum.masked_fill(~mask, float("-inf"))

    return torch.exp(x_segsum)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model with Triton-accelerated segsum.

        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
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

    def segsum_exp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute exp(segsum(x)) along the last dimension.

        Uses Triton kernel on CUDA tensors, falls back to a naive PyTorch
        implementation otherwise.
        """
        if x.is_cuda:
            return segsum_exp_triton(x)
        else:
            return segsum_exp_naive(x)

    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation.

        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y
        """
        B, L, H, P = X.shape
        C = self.block_len
        assert L % C == 0
        num_chunks = L // C

        # 0. Reshape into blocks/chunks
        # X: (b, (c l), h, p) -> (b, c, l, h, p)
        X_blocks = X.view(B, num_chunks, C, H, P)

        # A: (b, (c l), h) -> (b, c, l, h)
        A_blocks0 = self.A.view(B, num_chunks, C, H)
        # B, C: (b, (c l), h, n) -> (b, c, l, h, n)
        B_blocks = self.B.view(B, num_chunks, C, H, self.d_state)
        C_blocks = self.C.view(B, num_chunks, C, H, self.d_state)

        # A_blocks: (b, h, c, l)
        A_blocks = A_blocks0.permute(0, 3, 1, 2).contiguous()
        A_cumsum = torch.cumsum(A_blocks, dim=-1)

        # 1. Compute diagonal block outputs
        # L = exp(segsum(A_blocks))  -- via Triton-accelerated segsum_exp
        L = self.segsum_exp(A_blocks)  # (b, h, c, l, l)

        # Original einsum:
        # "bclhn,bcshn,bhcls,bcshp->bclhp"
        # Shapes:
        #   C_blocks: (b, c, l, h, n)
        #   B_blocks: (b, c, s, h, n)  (s == l)
        #   L:        (b, h, c, l, s)
        #   X_blocks: (b, c, s, h, p)
        Y_diag = torch.einsum(
            "bclhn,bcshn,bhcls,bcshp->bclhp",
            C_blocks,
            B_blocks,
            L,
            X_blocks,
        )

        # 2. Compute intra-chunk states
        decay_states = torch.exp(A_cumsum[:, :, :, -1:].sub(A_cumsum))
        # "bclhn,bhcl,bclhp->bchpn"
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

        # Pad last element of A_cumsum across chunks and apply segsum + exp
        # A_cumsum[:, :, :, -1]: (b, h, c)
        decay_input = torch.nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))
        # decay_chunk: (b, h, c+1, c+1)
        decay_chunk = self.segsum_exp(decay_input)

        # "bhzc,bchpn->bzhpn"
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        # 4. Compute state-to-output conversion
        state_decay_out = torch.exp(A_cumsum)
        # "bclhn,bchpn,bhcl->bclhp"
        Y_off = torch.einsum(
            "bclhn,bchpn,bhcl->bclhp",
            C_blocks,
            states,
            state_decay_out,
        )

        # Combine diagonal and off-diagonal terms
        Y = Y_diag + Y_off  # (b, c, l, h, p)
        Y = Y.reshape(B, num_chunks * C, H, P)

        return Y
