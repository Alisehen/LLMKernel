# complete ModelNew code with optimized Triton kernels

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B (+ bias) where:
      A: [M, K]
      B: [K, N]
      bias: [N]
      C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(c_ptrs, acc, mask=mask_c)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None, out: torch.Tensor = None):
    """
    High-performance linear layer using Triton:
      y = x @ weight + bias
    where:
      x:      [M, K]
      weight: [K, N]
      bias:   [N]
      out:    [M, N] (optional preallocated)
    """
    assert x.dim() == 2, "x must be 2D (M, K)"
    assert weight.dim() == 2, "weight must be 2D (K, N)"
    M, K = x.shape
    Kw, N = weight.shape
    assert Kw == K, f"Incompatible shapes: x: ({M}, {K}), weight: ({Kw}, {N})"

    if out is None:
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    else:
        assert out.shape == (M, N), "out must have shape (M, N)"

    has_bias = bias is not None
    if has_bias:
        assert bias.shape[0] == N, "bias must have shape (N,)"

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_kernel[grid](
        x, weight, bias if has_bias else x, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def gru_layer_kernel(
    gi_ptr,        # [T, B, 3H] precomputed input-side gates
    h_ptr,         # [B, H] working hidden state buffer (updated in-place)
    w_hh_ptr,      # [H, 3H] recurrent weights
    b_hh_ptr,      # [3H] recurrent bias (or dummy)
    out_ptr,       # [T, B, H] output hidden states
    T, B, H,
    stride_gi_t, stride_gi_b, stride_gi_c,
    stride_h_b, stride_h_c,
    stride_w_k, stride_w_c,
    stride_out_t, stride_out_b, stride_out_c,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Persistent GRU layer kernel for one direction:
      - Time loop is inside the kernel.
      - Input-side gates gi_t are precomputed (outside) and passed in.
      - This kernel computes recurrent-side gates, non-linearities, and state
        updates, and writes the full sequence of hidden states to out_ptr.
    """
    pid_m = tl.program_id(0)  # along batch dimension
    pid_n = tl.program_id(1)  # along hidden dimension

    offs_b = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_b = offs_b < B
    mask_n = offs_n < H
    mask_bn = mask_b[:, None] & mask_n[None, :]

    # Precompute offsets that are time-invariant
    # Pointers into the hidden-state buffer (B, H)
    h_base_ptrs = h_ptr + offs_b[:, None] * stride_h_b + offs_n[None, :] * stride_h_c

    # Bias slices for the three gates (size H each)
    if HAS_BIAS:
        b_r = tl.load(b_hh_ptr + offs_n, mask=mask_n, other=0.0)
        b_z = tl.load(b_hh_ptr + (offs_n + H), mask=mask_n, other=0.0)
        b_n = tl.load(b_hh_ptr + (offs_n + 2 * H), mask=mask_n, other=0.0)
    else:
        b_r = tl.zeros((BLOCK_N,), dtype=tl.float32)
        b_z = tl.zeros((BLOCK_N,), dtype=tl.float32)
        b_n = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Constants for gate column offsets in gi and W_hh
    col_offset_z = H
    col_offset_n = 2 * H

    offs_k = tl.arange(0, BLOCK_K)

    # Time loop (persistent across timesteps)
    for t in range(0, T):
        # ---- Recurrent-side GEMMs: h @ W_{hh} for r, z, n gates ----
        acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_n = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over K dimension (hidden size) to compute GEMMs
        for k in range(0, H, BLOCK_K):
            k_idx = k + offs_k
            k_mask = k_idx < H

            # Load a tile of current hidden state h: [BLOCK_M, BLOCK_K]
            h_tile_ptrs = h_ptr + offs_b[:, None] * stride_h_b + k_idx[None, :] * stride_h_c
            h_tile_mask = mask_b[:, None] & k_mask[None, :]
            h_tile = tl.load(h_tile_ptrs, mask=h_tile_mask, other=0.0)

            # Load corresponding weight tiles for r, z, n gates
            # W_hh layout: [H, 3H] with strides (stride_w_k, stride_w_c)
            w_base_ptrs = w_hh_ptr + k_idx[:, None] * stride_w_k + offs_n[None, :] * stride_w_c
            w_mask = k_mask[:, None] & mask_n[None, :]

            w_r = tl.load(w_base_ptrs, mask=w_mask, other=0.0)
            w_z = tl.load(w_base_ptrs + col_offset_z * stride_w_c, mask=w_mask, other=0.0)
            w_n = tl.load(w_base_ptrs + col_offset_n * stride_w_c, mask=w_mask, other=0.0)

            acc_r += tl.dot(h_tile, w_r, allow_tf32=True)
            acc_z += tl.dot(h_tile, w_z, allow_tf32=True)
            acc_n += tl.dot(h_tile, w_n, allow_tf32=True)

        # Add recurrent bias
        acc_r += b_r[None, :]
        acc_z += b_z[None, :]
        # acc_n bias handled separately through b_n and gi_n

        # ---- Input-side gates (precomputed) ----
        gi_t_ptr = gi_ptr + t * stride_gi_t

        gi_r_ptrs = gi_t_ptr + offs_b[:, None] * stride_gi_b + offs_n[None, :] * stride_gi_c
        gi_z_ptrs = gi_t_ptr + offs_b[:, None] * stride_gi_b + (offs_n[None, :] + col_offset_z) * stride_gi_c
        gi_n_ptrs = gi_t_ptr + offs_b[:, None] * stride_gi_b + (offs_n[None, :] + col_offset_n) * stride_gi_c

        gi_r = tl.load(gi_r_ptrs, mask=mask_bn, other=0.0)
        gi_z = tl.load(gi_z_ptrs, mask=mask_bn, other=0.0)
        gi_n = tl.load(gi_n_ptrs, mask=mask_bn, other=0.0)

        # ---- Gate activations ----
        # r = sigmoid(gi_r + gh_r)
        # z = sigmoid(gi_z + gh_z)
        gates_r = gi_r + acc_r
        gates_z = gi_z + acc_z

        r = 1.0 / (1.0 + tl.exp(-gates_r))
        z = 1.0 / (1.0 + tl.exp(-gates_z))

        # n = tanh(gi_n + r * gh_n + b_n)
        gh_n = acc_n + b_n[None, :]
        gates_n = gi_n + r * gh_n
        n_val = tl.tanh(gates_n)

        # Load previous hidden state h_prev for this tile (for update)
        h_prev = tl.load(h_base_ptrs, mask=mask_bn, other=0.0)

        # h = n + z * (h_prev - n)
        h_new = n_val + z * (h_prev - n_val)

        # Store updated hidden state back to working buffer
        tl.store(h_base_ptrs, h_new, mask=mask_bn)

        # Also write to output sequence at timestep t
        out_t_ptrs = out_ptr + t * stride_out_t + offs_b[:, None] * stride_out_b + offs_n[None, :] * stride_out_c
        tl.store(out_t_ptrs, h_new, mask=mask_bn)


def gru_layer_persistent(gi: torch.Tensor,
                         w_hh: torch.Tensor,
                         b_hh: torch.Tensor,
                         h0: torch.Tensor) -> torch.Tensor:
    """
    Run one GRU layer direction with a persistent Triton kernel.

    Args:
      gi:  [T, B, 3H] precomputed input-side gates for this direction
      w_hh: [H, 3H] recurrent weights
      b_hh: [3H] recurrent bias (or None)
      h0: [B, H] initial hidden state for this direction

    Returns:
      out: [T, B, H] full sequence of hidden states
    """
    assert gi.dim() == 3
    T, B, threeH = gi.shape
    H = h0.shape[1]
    assert threeH == 3 * H
    assert w_hh.shape == (H, 3 * H)

    device = gi.device
    dtype = gi.dtype

    # Working hidden buffer, initialized from h0
    h_buf = h0.contiguous().clone()

    out = torch.empty((T, B, H), device=device, dtype=dtype)

    has_bias = b_hh is not None
    if has_bias:
        assert b_hh.shape[0] == 3 * H

    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_M"]),
        triton.cdiv(H, META["BLOCK_N"]),
    )

    gru_layer_kernel[grid](
        gi, h_buf, w_hh, b_hh if has_bias else gi, out,
        T, B, H,
        gi.stride(0), gi.stride(1), gi.stride(2),
        h_buf.stride(0), h_buf.stride(1),
        w_hh.stride(0), w_hh.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        HAS_BIAS=has_bias,
        BLOCK_M=32, BLOCK_N=64, BLOCK_K=32,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        Custom multi-layer bidirectional GRU implemented with a persistent
        Triton kernel for the recurrent computations.

        Parameters mirror the original Model's GRU.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2  # bidirectional=True

        # Parameters are stored in (in_features, 3 * hidden_size) format so
        # that matmul can be done as x @ W efficiently (no transpose needed).
        self.weight_ih_f = nn.ParameterList()
        self.weight_hh_f = nn.ParameterList()
        self.weight_ih_b = nn.ParameterList()
        self.weight_hh_b = nn.ParameterList()

        if self.bias:
            self.bias_ih_f = nn.ParameterList()
            self.bias_hh_f = nn.ParameterList()
            self.bias_ih_b = nn.ParameterList()
            self.bias_hh_b = nn.ParameterList()
        else:
            self.bias_ih_f = None
            self.bias_hh_f = None
            self.bias_ih_b = None
            self.bias_hh_b = None

        for layer in range(num_layers):
            if layer == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size * self.num_directions

            # Forward direction weights
            w_ih_f = nn.Parameter(torch.empty(layer_input_size, 3 * hidden_size))
            w_hh_f = nn.Parameter(torch.empty(hidden_size, 3 * hidden_size))
            # Backward direction weights
            w_ih_b = nn.Parameter(torch.empty(layer_input_size, 3 * hidden_size))
            w_hh_b = nn.Parameter(torch.empty(hidden_size, 3 * hidden_size))

            # Initialization similar to common RNN practice
            nn.init.xavier_uniform_(w_ih_f)
            nn.init.xavier_uniform_(w_ih_b)
            nn.init.orthogonal_(w_hh_f)
            nn.init.orthogonal_(w_hh_b)

            self.weight_ih_f.append(w_ih_f)
            self.weight_hh_f.append(w_hh_f)
            self.weight_ih_b.append(w_ih_b)
            self.weight_hh_b.append(w_hh_b)

            if self.bias:
                b_ih_f = nn.Parameter(torch.zeros(3 * hidden_size))
                b_hh_f = nn.Parameter(torch.zeros(3 * hidden_size))
                b_ih_b = nn.Parameter(torch.zeros(3 * hidden_size))
                b_hh_b = nn.Parameter(torch.zeros(3 * hidden_size))

                self.bias_ih_f.append(b_ih_f)
                self.bias_hh_f.append(b_hh_f)
                self.bias_ih_b.append(b_ih_b)
                self.bias_hh_b.append(b_hh_b)

    def forward(self, x, h0):
        """
        x:  (seq_len, batch_size, input_size) if batch_first == False
             or (batch_size, seq_len, input_size) if batch_first == True
        h0: (num_layers * num_directions, batch_size, hidden_size)

        Returns:
          output: same as nn.GRU's output (we return only output, like original Model)
        """
        # Reorder to (seq_len, batch, feature) if necessary
        if self.batch_first:
            x = x.transpose(0, 1)  # (batch, seq, feat) -> (seq, batch, feat)

        seq_len, batch_size, _ = x.shape

        # If h0 is None, initialize to zeros
        if h0 is None:
            h0 = x.new_zeros(self.num_layers * self.num_directions,
                             batch_size, self.hidden_size)

        output = x
        final_h_states = []

        for layer in range(self.num_layers):
            layer_input_size = output.size(2)

            # Flatten sequence for input linear (GI) computations
            # Shape: (seq_len * batch_size, layer_input_size)
            inp2d = output.reshape(seq_len * batch_size, layer_input_size)

            # Forward direction parameters
            w_ih_f = self.weight_ih_f[layer]
            w_hh_f = self.weight_hh_f[layer]
            b_ih_f = self.bias_ih_f[layer] if self.bias else None
            b_hh_f = self.bias_hh_f[layer] if self.bias else None

            # Backward direction parameters
            w_ih_b = self.weight_ih_b[layer]
            w_hh_b = self.weight_hh_b[layer]
            b_ih_b = self.bias_ih_b[layer] if self.bias else None
            b_hh_b = self.bias_hh_b[layer] if self.bias else None

            # Precompute input-side linear transforms for all timesteps and both directions
            gi_f_flat = triton_linear(inp2d, w_ih_f, b_ih_f)  # (S*B, 3H)
            gi_b_flat = triton_linear(inp2d, w_ih_b, b_ih_b)  # (S*B, 3H)

            gi_f = gi_f_flat.reshape(seq_len, batch_size, 3 * self.hidden_size)
            gi_b = gi_b_flat.reshape(seq_len, batch_size, 3 * self.hidden_size)

            # ---- Forward direction (time increasing) ----
            h_f0 = h0[layer * self.num_directions + 0]  # (B, H)
            out_f = gru_layer_persistent(gi_f, w_hh_f, b_hh_f, h_f0)
            final_h_states.append(out_f[-1])

            # ---- Backward direction (time decreasing) ----
            # Reverse input-side gates along time, run the same kernel, then flip outputs back.
            h_b0 = h0[layer * self.num_directions + 1]  # (B, H)
            gi_b_rev = gi_b.flip(0).contiguous()
            out_b_rev = gru_layer_persistent(gi_b_rev, w_hh_b, b_hh_b, h_b0)
            out_b = out_b_rev.flip(0)
            # Final hidden state for backward direction corresponds to original t=0
            final_h_states.append(out_b_rev[-1])

            # Concatenate forward and backward outputs for this layer
            output = torch.cat([out_f, out_b], dim=2)  # (seq_len, batch, 2 * hidden)

        h_n = torch.stack(final_h_states, dim=0)  # (num_layers * num_directions, batch, hidden)

        if self.batch_first:
            output = output.transpose(0, 1)  # back to (batch, seq, feature)

        # Match original Model: return only output (not h_n)
        return output
