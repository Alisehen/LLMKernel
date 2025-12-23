import torch
import torch.nn as nn
import triton
import triton.language as tl


# ==========================
#  Triton helpers
# ==========================

@triton.jit
def _sigmoid(x):
    # Numerically-stable enough for GRU on fp16/fp32
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _tanh(x):
    # Express tanh via exp for Triton (no tl.tanh)
    e2x = tl.exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


# ==========================
#  Triton Kernels
# ==========================

@triton.autotune(
    configs=[
        # Balanced, good default
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        # More tiles in M
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        # More tiles in N
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute: C = A @ B + bias
    A: [M, K]
    B: [K, N]  (implemented via strides on a [N, K] stored tensor)
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

    k = 0
    while k < K:
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(tl.float32),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 64}, num_warps=2),
        triton.Config({'BLOCK_H': 128}, num_warps=4),
        triton.Config({'BLOCK_H': 256}, num_warps=4),
    ],
    key=['H'],
)
@triton.jit
def gru_cell_kernel(
    gate_x_ptr, gate_h_ptr, h_prev_ptr, h_new_ptr,
    B, H,
    stride_gxb, stride_gxh,
    stride_ghb, stride_ghh,
    stride_hb, stride_hh,
    BLOCK_H: tl.constexpr,
):
    """
    One GRU cell step for a full batch.

    gate_x: [B, 3H]  = x_t @ W_ih^T + b_ih
    gate_h: [B, 3H]  = h_prev @ W_hh^T + b_hh
    h_prev: [B, H]
    h_new:  [B, H]

    PyTorch GRU equations:
      i_r, i_z, i_n = gate_x.chunk(3, 1)
      h_r, h_z, h_n = gate_h.chunk(3, 1)

      r = sigmoid(i_r + h_r)
      z = sigmoid(i_z + h_z)
      n = tanh(i_n + r * h_n)

      h_new = (1 - z) * n + z * h_prev
    """
    pid_b = tl.program_id(0)
    pid_h_block = tl.program_id(1)

    offs_h = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # Row pointers
    gx_row = gate_x_ptr + pid_b * stride_gxb
    gh_row = gate_h_ptr + pid_b * stride_ghb
    hp_row = h_prev_ptr + pid_b * stride_hb
    hn_row = h_new_ptr + pid_b * stride_hb

    # Indices for gates
    offs_r = offs_h
    offs_z = offs_h + H
    offs_n = offs_h + 2 * H

    # ---- r gate ----
    i_r = tl.load(gx_row + offs_r * stride_gxh, mask=mask_h, other=0.0)
    h_r = tl.load(gh_row + offs_r * stride_ghh, mask=mask_h, other=0.0)
    r = i_r + h_r
    r = _sigmoid(r)

    # ---- z gate ----
    i_z = tl.load(gx_row + offs_z * stride_gxh, mask=mask_h, other=0.0)
    h_z = tl.load(gh_row + offs_z * stride_ghh, mask=mask_h, other=0.0)
    z = i_z + h_z
    z = _sigmoid(z)

    # ---- n gate ----
    i_n = tl.load(gx_row + offs_n * stride_gxh, mask=mask_h, other=0.0)
    h_n = tl.load(gh_row + offs_n * stride_ghh, mask=mask_h, other=0.0)
    n = i_n + r * h_n
    n = _tanh(n)

    # ---- final h ----
    h_prev = tl.load(hp_row + offs_h * stride_hh, mask=mask_h, other=0.0)
    h_new = (1.0 - z) * n + z * h_prev

    tl.store(hn_row + offs_h * stride_hh, h_new, mask=mask_h)


# ==========================
#  Python Wrappers
# ==========================

def linear_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    High-performance linear: y = x @ weight^T + bias
    x:      [M, K]
    weight: [3H, K]   (as in PyTorch GRU, out_features=3H, in_features=K)
    bias:   [3H]
    returns y: [M, 3H]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    M, K = x_contig.shape
    out_features = w_contig.shape[0]
    N = out_features

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_bias_kernel[grid](
        x_contig, w_contig, bias, y,
        M, N, K,
        x_contig.stride(0), x_contig.stride(1),
        # treat weight as [K, N] via strides on [N, K]
        w_contig.stride(1), w_contig.stride(0),
        y.stride(0), y.stride(1),
    )
    return y


def gru_cell_triton(
    gate_x: torch.Tensor,
    gate_h: torch.Tensor,
    h_prev: torch.Tensor,
) -> torch.Tensor:
    """
    Single GRU cell step on a batch using Triton.
    gate_x: [B, 3H] = x_t @ W_ih^T + b_ih
    gate_h: [B, 3H] = h_prev @ W_hh^T + b_hh
    h_prev: [B, H]
    returns h_new: [B, H]
    """
    assert gate_x.is_cuda and gate_h.is_cuda and h_prev.is_cuda
    B, threeH = gate_x.shape
    H = h_prev.shape[1]
    assert threeH == 3 * H, "gate_x second dim must be 3 * hidden_size"
    assert gate_h.shape == gate_x.shape

    gate_x_c = gate_x.contiguous()
    gate_h_c = gate_h.contiguous()
    h_prev_c = h_prev.contiguous()
    h_new = torch.empty_like(h_prev_c)

    grid = lambda META: (
        B,
        triton.cdiv(H, META["BLOCK_H"]),
    )

    gru_cell_kernel[grid](
        gate_x_c, gate_h_c, h_prev_c, h_new,
        B, H,
        gate_x_c.stride(0), gate_x_c.stride(1),
        gate_h_c.stride(0), gate_h_c.stride(1),
        h_prev_c.stride(0), h_prev_c.stride(1),
    )
    return h_new


# ==========================
#  Model with Triton GRU
# ==========================

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        Bidirectional multi-layer GRU implemented with Triton kernels.

        API matches:
          Model(input_size, hidden_size, num_layers=3, bias=True, batch_first=False)

        forward(x, h0) -> h_n
          h_n: [num_layers * num_directions, batch, hidden_size]
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2

        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        self.bias_ih = nn.ParameterList()
        self.bias_hh = nn.ParameterList()

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            for direction in range(self.num_directions):
                # W_ih: [3H, input_dim]
                w_ih = nn.Parameter(torch.empty(3 * hidden_size, layer_input_size))
                # W_hh: [3H, H]
                w_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
                nn.init.xavier_uniform_(w_ih)
                nn.init.orthogonal_(w_hh)

                self.weight_ih.append(w_ih)
                self.weight_hh.append(w_hh)

                if bias:
                    b_ih = nn.Parameter(torch.zeros(3 * hidden_size))
                    b_hh = nn.Parameter(torch.zeros(3 * hidden_size))
                else:
                    b_ih = nn.Parameter(torch.zeros(3 * hidden_size), requires_grad=False)
                    b_hh = nn.Parameter(torch.zeros(3 * hidden_size), requires_grad=False)

                self.bias_ih.append(b_ih)
                self.bias_hh.append(b_hh)

    def _gru_layer_direction(self, layer_input: torch.Tensor, h0_dir: torch.Tensor,
                             layer: int, direction: int):
        """
        Process one layer, one direction.

        layer_input: [T, B, input_dim]
        h0_dir:      [B, H]
        returns: (output_seq, h_last)
          output_seq: [T, B, H] in original time order
          h_last:     [B, H]
        """
        T, B, input_dim = layer_input.shape
        H = self.hidden_size
        device = layer_input.device
        dtype = layer_input.dtype

        if direction == 0:
            seq = layer_input
        else:
            # Backward: process reversed sequence
            seq = torch.flip(layer_input, [0])

        # Parameters for this layer/direction
        idx = layer * self.num_directions + direction
        W_ih = self.weight_ih[idx]
        W_hh = self.weight_hh[idx]
        b_ih = self.bias_ih[idx]
        b_hh = self.bias_hh[idx]

        # Precompute gate_x for all timesteps in one big GEMM: (T*B, input_dim) x W_ih^T
        seq_flat = seq.reshape(T * B, input_dim)
        gate_x_flat = linear_triton(seq_flat, W_ih, b_ih)  # [T*B, 3H]
        gate_x = gate_x_flat.view(T, B, 3 * H).contiguous()

        output_seq = torch.empty((T, B, H), device=device, dtype=dtype)
        h_prev = h0_dir

        for t in range(T):
            gate_x_t = gate_x[t]          # [B, 3H]
            gate_h_t = linear_triton(h_prev, W_hh, b_hh)  # [B, 3H]
            h_new = gru_cell_triton(gate_x_t, gate_h_t, h_prev)  # [B, H]
            output_seq[t] = h_new
            h_prev = h_new

        if direction == 1:
            # Back to original time order
            output_seq = torch.flip(output_seq, [0])

        return output_seq, h_prev

    def forward(self, x, h0):
        """
        x:  (seq_len, batch, input_size) if batch_first=False
            (batch, seq_len, input_size)  if batch_first=True

        h0: (num_layers * num_directions, batch, hidden_size)

        returns:
          h_n: (num_layers * num_directions, batch, hidden_size)
        """
        if self.batch_first:
            # (B, T, C) -> (T, B, C)
            x = x.transpose(0, 1)

        T, B, _ = x.shape
        H = self.hidden_size
        num_directions = self.num_directions

        # Input to the first layer
        layer_input = x
        h_n_list = []

        for layer in range(self.num_layers):
            # h0 for this layer: [2, B, H]
            h0_layer = h0[layer * num_directions:(layer + 1) * num_directions]

            # Forward direction
            out_f, h_last_f = self._gru_layer_direction(layer_input, h0_layer[0], layer, direction=0)
            # Backward direction
            out_b, h_last_b = self._gru_layer_direction(layer_input, h0_layer[1], layer, direction=1)

            # Prepare input for next layer: concat along feature dimension
            layer_input = torch.cat([out_f, out_b], dim=2)  # [T, B, 2H]

            h_n_list.append(h_last_f)
            h_n_list.append(h_last_b)

        h_n = torch.stack(h_n_list, dim=0)  # [num_layers * num_directions, B, H]

        return h_n
