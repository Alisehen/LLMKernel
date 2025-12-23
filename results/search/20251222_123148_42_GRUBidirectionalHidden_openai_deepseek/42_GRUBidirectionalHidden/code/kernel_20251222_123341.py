import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute: C = A @ B + bias
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

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def linear_bias_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: [M, K]
    weight: [N, K]  (same layout as nn.Linear.weight)
    bias: [N]
    returns: [M, N] = x @ weight.T + bias
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA"
    M, K = x.shape
    N = weight.shape[0]

    # We compute x @ weight.T as x [M,K] * weight_t [K,N]
    weight_t = weight.t().contiguous()
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_bias_kernel[grid](
        x, weight_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        Custom multi-layer bidirectional GRU implemented with Triton-accelerated linears.

        Only h_n is returned from forward(), matching the original Model.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # bidirectional=True
        self.batch_first = batch_first
        self.use_bias = bias

        # We store parameters for (layer, direction) in a flat list:
        # index = layer * num_directions + direction
        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        self.bias_ih = nn.ParameterList()
        self.bias_hh = nn.ParameterList()

        for layer in range(num_layers):
            if layer == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size * self.num_directions

            for direction in range(self.num_directions):
                # Input-to-hidden weights: [3*H, input_dim]
                w_ih = nn.Parameter(torch.empty(3 * hidden_size, layer_input_size))
                # Hidden-to-hidden weights: [3*H, H]
                w_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))

                if self.use_bias:
                    b_ih = nn.Parameter(torch.empty(3 * hidden_size))
                    b_hh = nn.Parameter(torch.empty(3 * hidden_size))
                else:
                    b_ih = nn.Parameter(torch.zeros(3 * hidden_size), requires_grad=False)
                    b_hh = nn.Parameter(torch.zeros(3 * hidden_size), requires_grad=False)

                self.weight_ih.append(w_ih)
                self.weight_hh.append(w_hh)
                self.bias_ih.append(b_ih)
                self.bias_hh.append(b_hh)

        self.reset_parameters()

    def reset_parameters(self):
        # Simple, reasonable initialization
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name and param.requires_grad:
                nn.init.zeros_(param)

    def _gru_layer(
        self,
        x: torch.Tensor,
        h0_layer: torch.Tensor,
        layer_idx: int,
    ):
        """
        Process a single GRU layer (both directions).

        x: [T, B, input_dim]
        h0_layer: [num_directions, B, H]
        returns:
          output: [T, B, num_directions * H]
          h_n_layer: [num_directions, B, H]
        """
        T, B, _ = x.shape
        H = self.hidden_size
        num_directions = self.num_directions

        layer_outputs = []
        final_h_list = []

        flat_input = x.reshape(T * B, -1)

        for direction in range(num_directions):
            param_idx = layer_idx * num_directions + direction

            w_ih = self.weight_ih[param_idx]
            w_hh = self.weight_hh[param_idx]
            b_ih = self.bias_ih[param_idx]
            b_hh = self.bias_hh[param_idx]

            # Precompute input-to-hidden gates for the whole sequence with one Triton GEMM
            gates_ih = linear_bias_triton(flat_input, w_ih, b_ih)
            gates_ih = gates_ih.reshape(T, B, 3 * H)

            h_prev = h0_layer[direction]  # [B, H]
            out_dir = torch.empty(T, B, H, device=x.device, dtype=x.dtype)

            if direction == 0:
                time_iter = range(T)  # forward
            else:
                time_iter = range(T - 1, -1, -1)  # backward

            for t in time_iter:
                gi = gates_ih[t]  # [B, 3H]

                # Hidden-to-hidden gates (per time step) with Triton GEMM
                gh = linear_bias_triton(h_prev, w_hh, b_hh)  # [B, 3H]

                gi_r, gi_z, gi_n = gi.split(H, dim=1)
                gh_r, gh_z, gh_n = gh.split(H, dim=1)

                r = torch.sigmoid(gi_r + gh_r)
                z = torch.sigmoid(gi_z + gh_z)
                n = torch.tanh(gi_n + r * gh_n)

                h_new = (1.0 - z) * n + z * h_prev
                h_prev = h_new

                if direction == 0:
                    out_dir[t] = h_new
                else:
                    # backward direction writes in original time order
                    out_dir[t] = h_new

            layer_outputs.append(out_dir)
            final_h_list.append(h_prev)

        # Concatenate directions on feature dimension
        output = torch.cat(layer_outputs, dim=2)  # [T, B, 2H]
        h_n_layer = torch.stack(final_h_list, dim=0)  # [2, B, H]
        return output, h_n_layer

    def forward(self, x, h0=None):
        """
        x: (seq_len, batch, input_size) if batch_first=False
           (batch, seq_len, input_size) if batch_first=True
        h0: (num_layers * num_directions, batch, hidden_size)
        returns: h_n (num_layers * num_directions, batch, hidden_size)
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # [T, B, C]

        T, B, _ = x.shape

        if h0 is None:
            h0 = x.new_zeros(self.num_layers * self.num_directions, B, self.hidden_size)

        assert h0.shape[0] == self.num_layers * self.num_directions
        assert h0.shape[1] == B
        assert h0.shape[2] == self.hidden_size

        current_x = x
        h_n_layers = []

        for layer in range(self.num_layers):
            # h0 for this layer: [num_directions, B, H]
            h0_layer = h0[layer * self.num_directions: (layer + 1) * self.num_directions]
            current_x, h_n_layer = self._gru_layer(current_x, h0_layer, layer)
            h_n_layers.append(h_n_layer)

        h_n = torch.cat(h_n_layers, dim=0)  # [num_layers * num_directions, B, H]
        return h_n
