import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------
# Low-level Triton kernels
# -------------------------

@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    High-performance GEMM + bias:
      y[M, N] = x[M, K] @ w[K, N] + b[N]

    - Accumulates in fp32, allows TF32 when inputs are fp32.
    - Single 2D grid over output tiles (M, N).
    - Bias add shares the same (offs_m, offs_n, mask) as the matmul output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for first K tile
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k

        # Masks for this K-block
        k_mask = k_idx < K
        x_mask = (offs_m[:, None] < M) & k_mask[None, :]
        w_mask = k_mask[:, None] & (offs_n[None, :] < N)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Tensor Core friendly dot
        acc += tl.dot(x, w, allow_tf32=True)

        # Advance pointers for next K tile
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Output mask over (M, N) tile
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Bias: broadcast along M, same mask as output
    # b[N] -> (1, N) -> (BLOCK_M, BLOCK_N) via broadcasting
    bias_ptrs = b_ptr + offs_n[None, :]
    bias = tl.load(bias_ptrs, mask=y_mask, other=0.0)
    acc += bias

    # Store result (implicit cast to y dtype)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def lstm_pointwise_kernel(
    gates_ptr, c_prev_ptr, h_new_ptr, c_new_ptr,
    M, H,
    stride_gm, stride_gn,
    stride_cm, stride_cn,
    stride_hm, stride_hn,
    stride_cnm, stride_cnn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Pointwise LSTM update for one timestep:
      gates: [M, 4H] = [i, f, g, o] pre-activations (row-major)
      c_prev: [M, H]

      c_new = f * c_prev + i * g
      h_new = o * tanh(c_new)

    All elementwise ops share the same grid, offsets, and mask over [M, H].
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Unified mask over [M, H]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < H)

    # Base pointers for i gate (gates[:, :H])
    gates_i_ptrs = gates_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn

    # Load 4 gates: i, f, g, o
    gi = tl.load(gates_i_ptrs, mask=mask, other=0.0)
    gf = tl.load(gates_i_ptrs + H * stride_gn, mask=mask, other=0.0)
    gg = tl.load(gates_i_ptrs + 2 * H * stride_gn, mask=mask, other=0.0)
    go = tl.load(gates_i_ptrs + 3 * H * stride_gn, mask=mask, other=0.0)

    # Previous cell state
    c_prev_ptrs = c_prev_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_prev = tl.load(c_prev_ptrs, mask=mask, other=0.0)

    one = 1.0

    # sigmoid(x) = 1 / (1 + exp(-x))
    i = one / (one + tl.exp(-gi))
    f = one / (one + tl.exp(-gf))
    o = one / (one + tl.exp(-go))

    # tanh(g): (exp(2x) - 1) / (exp(2x) + 1)
    x2 = 2.0 * gg
    ex2 = tl.exp(x2)
    g = (ex2 - 1.0) / (ex2 + 1.0)

    # Cell update
    c_new = f * c_prev + i * g

    # tanh(c_new)
    c2 = 2.0 * c_new
    ec2 = tl.exp(c2)
    tanh_c = (ec2 - 1.0) / (ec2 + 1.0)

    h_new = o * tanh_c

    # Stores — same offsets & mask
    h_new_ptrs = h_new_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn
    c_new_ptrs = c_new_ptr + offs_m[:, None] * stride_cnm + offs_n[None, :] * stride_cnn

    tl.store(h_new_ptrs, h_new, mask=mask)
    tl.store(c_new_ptrs, c_new, mask=mask)


# -------------------------
# Wrapper functions
# -------------------------

def triton_linear(x, w, b, out=None):
    """
    High-performance linear: y = x @ w + b
      x: [M, K]
      w: [K, N]
      b: [N]
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda
    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w, f"Incompatible shapes: x={x.shape}, w={w.shape}"

    if (
        out is None
        or out.shape != (M, N)
        or out.dtype != x.dtype
        or out.device != x.device
    ):
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    # Tile tuned for Ada (4090) for typical LSTM dimensions
    linear_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64,   # power-of-2
        BLOCK_N=128,  # power-of-2
        BLOCK_K=32,   # power-of-2
        num_warps=8,
        num_stages=3,
    )
    return out


def lstm_pointwise(gates, c_prev, h_out=None, c_out=None):
    """
    LSTM pointwise step for one timestep:
      gates: [M, 4H]
      c_prev: [M, H]
    Returns (h_new, c_new): both [M, H]
    """
    assert gates.is_cuda and c_prev.is_cuda
    M, fourH = gates.shape
    H = fourH // 4
    assert c_prev.shape == (M, H)

    if (
        h_out is None
        or h_out.shape != (M, H)
        or h_out.dtype != gates.dtype
        or h_out.device != gates.device
    ):
        h_out = torch.empty((M, H), device=gates.device, dtype=gates.dtype)
    if (
        c_out is None
        or c_out.shape != (M, H)
        or c_out.dtype != gates.dtype
        or c_out.device != gates.device
    ):
        c_out = torch.empty((M, H), device=gates.device, dtype=gates.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(H, meta["BLOCK_N"]),
        )

    lstm_pointwise_kernel[grid](
        gates, c_prev, h_out, c_out,
        M, H,
        gates.stride(0), gates.stride(1),
        c_prev.stride(0), c_prev.stride(1),
        h_out.stride(0), h_out.stride(1),
        c_out.stride(0), c_out.stride(1),
        BLOCK_M=32,   # power-of-2
        BLOCK_N=64,   # power-of-2
        num_warps=4,
        num_stages=2,
    )
    return h_out, c_out


# -------------------------
# High-level ModelNew
# -------------------------

class ModelNew(nn.Module):
    """
    Triton-optimized reimplementation of the original LSTM model.

    Uses nn.LSTM and nn.Linear only as parameter containers so that
    state_dict() is compatible with the original Model. The actual
    computations are done via custom Triton kernels.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        # Parameter containers with identical structure to original Model
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        """
        x:  [batch, seq_len, input_size]
        h0: [num_layers, batch, hidden_size]
        c0: [num_layers, batch, hidden_size]

        Returns final cell state c_n (same as original: state[1]).
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels."
        batch_size, seq_len, _ = x.shape
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        dropout_p = self.lstm.dropout

        # Current inputs to each layer (batch, seq_len, features)
        layer_input = x

        h_n_list = []
        c_n_list = []

        for layer in range(num_layers):
            # LSTM parameters for this layer
            weight_ih = getattr(self.lstm, f"weight_ih_l{layer}")  # [4H, input_size or H]
            weight_hh = getattr(self.lstm, f"weight_hh_l{layer}")  # [4H, H]
            bias_ih = getattr(self.lstm, f"bias_ih_l{layer}")      # [4H]
            bias_hh = getattr(self.lstm, f"bias_hh_l{layer}")      # [4H]

            D_in = layer_input.size(2)
            H = hidden_size

            # Combine weights and biases: gates = x*W_ih^T + h*W_hh^T + b_ih + b_hh
            # W_cat: [4H, D_in + H] -> W_cat_t: [D_in + H, 4H]
            W_cat = torch.cat([weight_ih[:, :D_in], weight_hh], dim=1)
            W_cat_t = W_cat.t().contiguous()
            bias = (bias_ih + bias_hh).contiguous()

            # Initial hidden and cell states for this layer: [batch, H]
            h_t = h0[layer]
            c_t = c0[layer]

            # Reused buffers across timesteps to avoid reallocations
            layer_output = torch.empty(
                (batch_size, seq_len, H), device=x.device, dtype=x.dtype
            )
            gates_buf = torch.empty(
                (batch_size, 4 * H), device=x.device, dtype=x.dtype
            )
            h_buf = torch.empty(
                (batch_size, H), device=x.device, dtype=x.dtype
            )
            c_buf = torch.empty(
                (batch_size, H), device=x.device, dtype=x.dtype
            )

            for t in range(seq_len):
                x_t = layer_input[:, t, :]  # [batch, D_in]

                # Concatenate input and previous hidden: [batch, D_in + H]
                cat_input = torch.cat([x_t, h_t], dim=1)

                # Linear for 4 gates: [batch, 4H]
                gates_buf = triton_linear(cat_input, W_cat_t, bias, out=gates_buf)

                # Pointwise LSTM nonlinearity
                h_buf, c_buf = lstm_pointwise(gates_buf, c_t, h_out=h_buf, c_out=c_buf)

                # Save output and update states
                layer_output[:, t, :] = h_buf
                h_t = h_buf
                c_t = c_buf

            # Save final states for this layer
            h_n_list.append(h_t)
            c_n_list.append(c_t)

            # Dropout on outputs of this layer except the last
            if (layer < num_layers - 1) and (dropout_p > 0.0) and self.training:
                layer_input = torch.nn.functional.dropout(
                    layer_output, p=dropout_p, training=True
                )
            else:
                layer_input = layer_output

        # Final output sequence from top layer
        out = layer_input  # [batch, seq_len, hidden_size]

        # Decode last time step using Triton linear
        last_timestep = out[:, -1, :]  # [batch, hidden_size]
        fc_weight_t = self.fc.weight.t().contiguous()  # [hidden_size, output_size]
        _ = triton_linear(last_timestep, fc_weight_t, self.fc.bias)

        # Stack final states to match nn.LSTM API: [num_layers, batch, hidden_size]
        c_n = torch.stack(c_n_list, dim=0)
        return c_n
