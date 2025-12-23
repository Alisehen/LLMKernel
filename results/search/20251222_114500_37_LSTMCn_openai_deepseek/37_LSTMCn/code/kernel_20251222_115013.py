import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for tiles along M (rows) and N (cols)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows and columns in the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to tiles of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] + k < K

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask.T & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs, acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def lstm_pointwise_kernel(
    gates_x_ptr, gates_h_ptr,
    c_ptr, h_ptr,
    B, H,
    stride_gxb, stride_gxg,
    stride_ghb, stride_ghg,
    stride_cb, stride_ch,
    stride_hb, stride_hh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Tile identifiers
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # hidden dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    idx_b = offs_m[:, None]
    idx_h = offs_n[None, :]

    mask = (idx_b < B) & (idx_h < H)

    # Load gate pre-activations from x and h projections
    # Layout gates: [i | f | g | o] along last dimension (size 4H)
    # gates_x shape: (B, 4H)
    # gates_h shape: (B, 4H)
    i_x = tl.load(
        gates_x_ptr + idx_b * stride_gxb + idx_h * stride_gxg,
        mask=mask,
        other=0.0,
    )
    f_x = tl.load(
        gates_x_ptr + idx_b * stride_gxb + (idx_h + H) * stride_gxg,
        mask=mask,
        other=0.0,
    )
    g_x = tl.load(
        gates_x_ptr + idx_b * stride_gxb + (idx_h + 2 * H) * stride_gxg,
        mask=mask,
        other=0.0,
    )
    o_x = tl.load(
        gates_x_ptr + idx_b * stride_gxb + (idx_h + 3 * H) * stride_gxg,
        mask=mask,
        other=0.0,
    )

    i_h = tl.load(
        gates_h_ptr + idx_b * stride_ghb + idx_h * stride_ghg,
        mask=mask,
        other=0.0,
    )
    f_h = tl.load(
        gates_h_ptr + idx_b * stride_ghb + (idx_h + H) * stride_ghg,
        mask=mask,
        other=0.0,
    )
    g_h = tl.load(
        gates_h_ptr + idx_b * stride_ghb + (idx_h + 2 * H) * stride_ghg,
        mask=mask,
        other=0.0,
    )
    o_h = tl.load(
        gates_h_ptr + idx_b * stride_ghb + (idx_h + 3 * H) * stride_ghg,
        mask=mask,
        other=0.0,
    )

    # Combine contributions
    i = i_x + i_h
    f = f_x + f_h
    g = g_x + g_h
    o = o_x + o_h

    # Sigmoid: 1 / (1 + exp(-x))
    i = 1.0 / (1.0 + tl.exp(-i))
    f = 1.0 / (1.0 + tl.exp(-f))
    o = 1.0 / (1.0 + tl.exp(-o))

    # Tanh via definition: (exp(2x)-1)/(exp(2x)+1)
    two_g = 2.0 * g
    exp_two_g = tl.exp(two_g)
    g = (exp_two_g - 1.0) / (exp_two_g + 1.0)

    # Load previous cell state
    c_prev = tl.load(
        c_ptr + idx_b * stride_cb + idx_h * stride_ch,
        mask=mask,
        other=0.0,
    )

    # c_t = f * c_{t-1} + i * g
    c_new = f * c_prev + i * g

    # h_t = o * tanh(c_t)
    two_c = 2.0 * c_new
    exp_two_c = tl.exp(two_c)
    tanh_c = (exp_two_c - 1.0) / (exp_two_c + 1.0)
    h_new = o * tanh_c

    # Store back (in-place update of c and h)
    tl.store(
        c_ptr + idx_b * stride_cb + idx_h * stride_ch,
        c_new,
        mask=mask,
    )
    tl.store(
        h_ptr + idx_b * stride_hb + idx_h * stride_hh,
        h_new,
        mask=mask,
    )


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    High-performance linear layer: y = x @ weight^T + bias
    x: (M, K)
    weight: (N, K)
    bias: (N,)
    returns: (M, N)
    """
    assert x.dim() == 2
    assert weight.dim() == 2
    assert bias.dim() == 1
    M, K = x.shape
    N, Kw = weight.shape
    assert Kw == K, "Incompatible shapes for matmul"

    # Output
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # We want B (for kernel) as shape (K, N)
    b_mat = weight.t().contiguous()

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    linear_kernel[grid](
        x, b_mat, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        b_mat.stride(0), b_mat.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return y


def triton_lstm_pointwise(gates_x: torch.Tensor,
                          gates_h: torch.Tensor,
                          c: torch.Tensor,
                          h: torch.Tensor):
    """
    Pointwise LSTM cell update using Triton.
    gates_x, gates_h: (B, 4H)
    c, h: (B, H) updated in-place to c_t, h_t
    """
    assert gates_x.shape == gates_h.shape
    B, fourH = gates_x.shape
    assert fourH % 4 == 0
    H = fourH // 4
    assert c.shape == (B, H)
    assert h.shape == (B, H)

    grid = lambda META: (
        triton.cdiv(B, META['BLOCK_M']),
        triton.cdiv(H, META['BLOCK_N']),
    )

    lstm_pointwise_kernel[grid](
        gates_x, gates_h,
        c, h,
        B, H,
        gates_x.stride(0), gates_x.stride(1),
        gates_h.stride(0), gates_h.stride(1),
        c.stride(0), c.stride(1),
        h.stride(0), h.stride(1),
        BLOCK_M=64, BLOCK_N=64,
    )


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Custom LSTM + Linear using high-performance Triton kernels.
        Matches the control flow of the original model:
        - Multi-layer unidirectional LSTM
        - Final Linear on the last time step of the last layer
        - forward() returns final cell states (c_n), shape (num_layers, batch, hidden_size)
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = float(dropout)

        # LSTM parameters (PyTorch-like layout but custom)
        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        self.bias_ih = nn.ParameterList()
        self.bias_hh = nn.ParameterList()

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size

            w_ih = nn.Parameter(torch.empty(4 * hidden_size, layer_input_size))
            w_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
            b_ih = nn.Parameter(torch.empty(4 * hidden_size))
            b_hh = nn.Parameter(torch.empty(4 * hidden_size))

            self.weight_ih.append(w_ih)
            self.weight_hh.append(w_hh)
            self.bias_ih.append(b_ih)
            self.bias_hh.append(b_hh)

        # Final fully-connected layer
        self.fc_weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.fc_bias = nn.Parameter(torch.empty(output_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize LSTM weights/biases similar to PyTorch defaults
        for layer in range(self.num_layers):
            w_ih = self.weight_ih[layer]
            w_hh = self.weight_hh[layer]
            b_ih = self.bias_ih[layer]
            b_hh = self.bias_hh[layer]

            # Kaiming uniform for weights
            nn.init.kaiming_uniform_(w_ih, a=math.sqrt(5))
            nn.init.kaiming_uniform_(w_hh, a=math.sqrt(5))

            # Biases to zero
            nn.init.zeros_(b_ih)
            nn.init.zeros_(b_hh)

        # Linear layer init like nn.Linear
        fan_in = self.fc_weight.size(1)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.fc_weight, -bound, bound)
        nn.init.uniform_(self.fc_bias, -bound, bound)

    def forward(self, x, h0, c0):
        """
        x:  (batch, seq_len, input_size)
        h0: (num_layers, batch, hidden_size)
        c0: (num_layers, batch, hidden_size)

        Returns:
            c_n: final cell states, shape (num_layers, batch, hidden_size)
        """
        # Ensure CUDA for Triton
        device = x.device
        assert x.is_cuda, "Triton kernels require CUDA tensors"

        batch_size, seq_len, _ = x.shape
        H = self.hidden_size

        # Current states per layer: each (B, H)
        h = [h0[layer].to(device=device).contiguous() for layer in range(self.num_layers)]
        c = [c0[layer].to(device=device).contiguous() for layer in range(self.num_layers)]

        # Temporary buffers for gate pre-activations per layer: (B, 4H)
        gates_x = [
            torch.empty((batch_size, 4 * H), device=device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]
        gates_h = [
            torch.empty((batch_size, 4 * H), device=device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]

        # Recurrent computation over time
        for t in range(seq_len):
            # Input to first layer at this time step
            layer_input = x[:, t, :]  # (B, input_size)

            for layer in range(self.num_layers):
                # Linear projections for gates: (B, 4H)
                wx = self.weight_ih[layer]
                wh = self.weight_hh[layer]
                bix = self.bias_ih[layer]
                bhh = self.bias_hh[layer]

                gx = triton_linear(layer_input, wx, bix)
                gh = triton_linear(h[layer], wh, bhh)

                gates_x[layer].copy_(gx)
                gates_h[layer].copy_(gh)

                # Pointwise LSTM cell update (in-place on h[layer], c[layer])
                triton_lstm_pointwise(gates_x[layer], gates_h[layer], c[layer], h[layer])

                # Output of this layer becomes input to next
                layer_input = h[layer]

        # Final hidden state of last layer (B, H)
        last_h = h[-1]

        # Final linear layer on last hidden state (not returned, just for parity)
        _ = triton_linear(last_h, self.fc_weight, self.fc_bias)

        # Stack final states to (num_layers, batch, hidden_size)
        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)

        # Match original Model: return cell states c_n
        return c_n


# math is used in reset_parameters
import math
