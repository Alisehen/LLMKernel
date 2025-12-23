import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------
# Optimized GEMM + bias kernel
# -----------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_bias_kernel(
    a_ptr,  # *A: [M, K]
    b_ptr,  # *B_t: [K, N]   (this is weight^T)
    bias_ptr,  # *bias: [N]
    c_ptr,  # *C: [M, N]

    M, N, K,
    stride_am, stride_ak,   # A: (row, col)
    stride_bk, stride_bn,   # B_t: (row=K, col=N)
    stride_cm, stride_cn,   # C: (row, col)

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B_t + bias

    A: [M, K]
    B_t: [K, N]   (transpose of [N, K])
    bias: [N]
    C: [M, N]

    Fused operations (matmul + bias) share the same output grid and
    (offs_m, offs_n) + boundary masks.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Global in-bounds flags for output tile
    in_bounds_m = offs_m < M
    in_bounds_n = offs_n < N

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Initial A/B pointers for this tile
    a_ptrs = a_ptr + (
        offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    )

    # K loop, pipelined via pointer arithmetic
    k = 0
    while k < K:
        k_remaining = K - k
        k_mask = offs_k < k_remaining  # [BLOCK_K]

        # All masks derived from the same global in-bounds conditions.
        a_mask = (in_bounds_m[:, None]) & (k_mask[None, :])
        b_mask = (k_mask[:, None]) & (in_bounds_n[None, :])

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply on the tile
        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Fused bias add: same (offs_n, in_bounds_n) as used for output.
    bias = tl.load(bias_ptr + offs_n, mask=in_bounds_n, other=0.0)
    acc += bias[None, :]

    # Write back C, using the shared output offsets and mask.
    c_ptrs = c_ptr + (
        offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    )
    c_mask = (in_bounds_m[:, None]) & (in_bounds_n[None, :])
    tl.store(c_ptrs, acc, mask=c_mask)


def _linear_bias_triton_impl(a: torch.Tensor,
                             b_t: torch.Tensor,
                             bias: torch.Tensor) -> torch.Tensor:
    """
    Low-level wrapper around the Triton kernel.

    a:   [M, K]
    b_t: [K, N]   (transpose of weight)
    bias: [N]
    returns: [M, N] = a @ b_t + bias
    """
    assert a.is_cuda and b_t.is_cuda and bias.is_cuda, "Tensors must be on CUDA"
    M, K = a.shape
    Kb, N = b_t.shape
    assert K == Kb, "Incompatible shapes for matmul"
    assert bias.shape[0] == N, "Bias must be length N"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_bias_kernel[grid](
        a, b_t, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_t.stride(0), b_t.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def linear_bias_triton(a: torch.Tensor,
                       weight: torch.Tensor,
                       bias: torch.Tensor) -> torch.Tensor:
    """
    Public wrapper kept API-compatible with the original version.

    a: [M, K]
    weight: [N, K]  (same layout as nn.Linear.weight)
    bias: [N]
    returns: [M, N] = a @ weight.T + bias
    """
    weight_t = weight.t().contiguous()
    return _linear_bias_triton_impl(a, weight_t, bias)


# -----------------------------
# GRU Model using Triton GEMMs
# -----------------------------

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

        # Parameters for (layer, direction) in a flat list:
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

        # Flatten time and batch for a single big input GEMM
        flat_input = x.reshape(T * B, -1)

        for direction in range(num_directions):
            param_idx = layer_idx * num_directions + direction

            w_ih = self.weight_ih[param_idx]  # [3H, input_dim]
            w_hh = self.weight_hh[param_idx]  # [3H, H]
            b_ih = self.bias_ih[param_idx]    # [3H]
            b_hh = self.bias_hh[param_idx]    # [3H]

            # Precompute transposed weights ONCE per layer+direction
            w_ih_t = w_ih.t().contiguous()  # [input_dim, 3H]
            w_hh_t = w_hh.t().contiguous()  # [H, 3H]

            # Precompute input-to-hidden gates for the whole sequence
            gates_ih = _linear_bias_triton_impl(flat_input, w_ih_t, b_ih)  # [T*B, 3H]
            gates_ih = gates_ih.view(T, B, 3 * H)

            h_prev = h0_layer[direction]  # [B, H]
            out_dir = torch.empty(T, B, H, device=x.device, dtype=x.dtype)

            if direction == 0:
                time_iter = range(T)  # forward
            else:
                time_iter = range(T - 1, -1, -1)  # backward

            for t in time_iter:
                gi = gates_ih[t]  # [B, 3H]

                # Hidden-to-hidden gates (per time step) with Triton GEMM
                gh = _linear_bias_triton_impl(h_prev, w_hh_t, b_hh)  # [B, 3H]

                gi_r, gi_z, gi_n = gi.split(H, dim=1)
                gh_r, gh_z, gh_n = gh.split(H, dim=1)

                r = torch.sigmoid(gi_r + gh_r)
                z = torch.sigmoid(gi_z + gh_z)
                n = torch.tanh(gi_n + r * gh_n)

                h_new = (1.0 - z) * n + z * h_prev
                h_prev = h_new

                # Write outputs in original time order for both directions
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
