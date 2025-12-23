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


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        Custom multi-layer bidirectional GRU implemented with Triton-accelerated
        linear layers for maximal performance.

        Parameters mirror the original Model's GRU.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2  # bidirectional=True

        # Parameters are stored in (in_features, 3 * hidden_size) format so that
        # matmul can be done as x @ W efficiently (no transpose needed).
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

            # Initialize weights similar to common RNN practice
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
        # Reorder dimensions to (seq_len, batch, feature) if necessary
        if self.batch_first:
            x = x.transpose(0, 1)  # (batch, seq, feat) -> (seq, batch, feat)

        seq_len, batch_size, _ = x.shape

        # If h0 is None, initialize to zeros
        if h0 is None:
            h0 = x.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)

        output = x
        final_h_states = []

        for layer in range(self.num_layers):
            layer_input_size = output.size(2)

            # Flatten sequence to one big matrix for input linear (GI) computations
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

            # Buffers for outputs
            out_f = output.new_empty(seq_len, batch_size, self.hidden_size)
            out_b = output.new_empty(seq_len, batch_size, self.hidden_size)

            # Initial hidden states for this layer (forward and backward)
            h_f = h0[layer * self.num_directions + 0]
            h_b = h0[layer * self.num_directions + 1]

            # Recurrent GEMM output buffers (reused every timestep)
            gh_f_buf = output.new_empty(batch_size, 3 * self.hidden_size)
            gh_b_buf = output.new_empty(batch_size, 3 * self.hidden_size)

            # ---- Forward direction ----
            for t in range(seq_len):
                gi_t = gi_f[t]  # (B, 3H)

                # gh = h_f @ W_hh_f + b_hh_f
                gh_t = triton_linear(h_f, w_hh_f, b_hh_f, out=gh_f_buf)

                gi_r, gi_z, gi_n = gi_t.chunk(3, dim=1)
                gh_r, gh_z, gh_n = gh_t.chunk(3, dim=1)

                r = torch.sigmoid(gi_r + gh_r)
                z = torch.sigmoid(gi_z + gh_z)
                n = torch.tanh(gi_n + r * gh_n)
                h_f = n + z * (h_f - n)

                out_f[t] = h_f

            final_h_states.append(h_f)

            # ---- Backward direction ----
            for t_inv in range(seq_len):
                t = seq_len - 1 - t_inv
                gi_t = gi_b[t]  # (B, 3H) corresponding to x_t in reverse order

                gh_t = triton_linear(h_b, w_hh_b, b_hh_b, out=gh_b_buf)

                gi_r, gi_z, gi_n = gi_t.chunk(3, dim=1)
                gh_r, gh_z, gh_n = gh_t.chunk(3, dim=1)

                r = torch.sigmoid(gi_r + gh_r)
                z = torch.sigmoid(gi_z + gh_z)
                n = torch.tanh(gi_n + r * gh_n)
                h_b = n + z * (h_b - n)

                out_b[t] = h_b

            final_h_states.append(h_b)

            # Concatenate forward and backward outputs for this layer
            output = torch.cat([out_f, out_b], dim=2)  # (seq_len, batch, 2 * hidden)

        h_n = torch.stack(final_h_states, dim=0)  # (num_layers * num_directions, batch, hidden)

        if self.batch_first:
            output = output.transpose(0, 1)  # back to (batch, seq, feature)

        # Match original Model: return only output (not h_n)
        return output
