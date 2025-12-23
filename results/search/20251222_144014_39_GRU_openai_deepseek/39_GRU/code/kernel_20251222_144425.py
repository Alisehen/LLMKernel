import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------
# GEMM + Bias kernel
# -----------------------
@triton.jit
def gru_gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute: C[M, N] = A[M, K] @ B[K, N] + bias[N]
    A: [M, K]
    B: [K, N]
    bias: [N]
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
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias[N]
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_out)


def gru_gemm_bias(x, weight, bias):
    """
    x:      [B, K]
    weight: [K, N]
    bias:   [N]
    returns [B, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    B, K = x.shape
    Kw, N = weight.shape
    assert Kw == K, "Input feature dim mismatch with weight"
    out = torch.empty((B, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    gru_gemm_bias_kernel[grid](
        x, weight, bias, out,
        B, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return out


# -----------------------
# GRU gate + state update kernel
# -----------------------
@triton.jit
def gru_gates_update_kernel(
    gate_x_ptr, gate_h_ptr,
    h_prev_ptr, h_new_ptr,
    B, H,                      # batch_size, hidden_size
    stride_gx0, stride_gx1,    # gate_x  [B, 3H]
    stride_gh0, stride_gh1,    # gate_h  [B, 3H]
    stride_hp0, stride_hp1,    # h_prev  [B, H]
    stride_hn0, stride_hn1,    # h_new   [B, H]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    gate_x: x_t * W_ih + b_ih   -> [B, 3H]
    gate_h: h_{t-1} * W_hh + b_hh -> [B, 3H]

    PyTorch GRU formulation ("reset-before"):
        i_r, i_z, i_n = gate_x.chunk(3, dim=1)
        h_r, h_z, h_n = gate_h.chunk(3, dim=1)

        r = sigmoid(i_r + h_r)
        z = sigmoid(i_z + h_z)
        n = tanh(i_n + r * h_n)

        h_t = (1 - z) * n + z * h_{t-1}
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_b = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # batch indices
    offs_h = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # hidden indices

    mask = (offs_b[:, None] < B) & (offs_h[None, :] < H)

    # pointers for each gate slice
    gx_r_ptrs = gate_x_ptr + offs_b[:, None] * stride_gx0 + offs_h[None, :] * stride_gx1
    gx_z_ptrs = gate_x_ptr + offs_b[:, None] * stride_gx0 + (offs_h[None, :] + H) * stride_gx1
    gx_n_ptrs = gate_x_ptr + offs_b[:, None] * stride_gx0 + (offs_h[None, :] + 2 * H) * stride_gx1

    gh_r_ptrs = gate_h_ptr + offs_b[:, None] * stride_gh0 + offs_h[None, :] * stride_gh1
    gh_z_ptrs = gate_h_ptr + offs_b[:, None] * stride_gh0 + (offs_h[None, :] + H) * stride_gh1
    gh_n_ptrs = gate_h_ptr + offs_b[:, None] * stride_gh0 + (offs_h[None, :] + 2 * H) * stride_gh1

    # load gates
    i_r = tl.load(gx_r_ptrs, mask=mask, other=0.0)
    i_z = tl.load(gx_z_ptrs, mask=mask, other=0.0)
    i_n = tl.load(gx_n_ptrs, mask=mask, other=0.0)

    h_r = tl.load(gh_r_ptrs, mask=mask, other=0.0)
    h_z = tl.load(gh_z_ptrs, mask=mask, other=0.0)
    h_n = tl.load(gh_n_ptrs, mask=mask, other=0.0)

    # previous hidden
    h_prev_ptrs = h_prev_ptr + offs_b[:, None] * stride_hp0 + offs_h[None, :] * stride_hp1
    h_prev = tl.load(h_prev_ptrs, mask=mask, other=0.0)

    # r = sigmoid(i_r + h_r)
    pre_r = i_r + h_r
    r = 1.0 / (1.0 + tl.exp(-pre_r))

    # z = sigmoid(i_z + h_z)
    pre_z = i_z + h_z
    z = 1.0 / (1.0 + tl.exp(-pre_z))

    # n = tanh(i_n + r * h_n)
    pre_n = i_n + r * h_n
    e = tl.exp(2.0 * pre_n)
    n = (e - 1.0) / (e + 1.0)

    # h_t = (1 - z) * n + z * h_prev
    h_t = (1.0 - z) * n + z * h_prev

    h_new_ptrs = h_new_ptr + offs_b[:, None] * stride_hn0 + offs_h[None, :] * stride_hn1
    tl.store(h_new_ptrs, h_t, mask=mask)


def gru_gates_update(gate_x, gate_h, h_prev):
    """
    gate_x: [B, 3H]
    gate_h: [B, 3H]
    h_prev: [B, H]
    returns h_new: [B, H]
    """
    assert gate_x.is_cuda and gate_h.is_cuda and h_prev.is_cuda
    B, threeH = gate_x.shape
    B2, threeH2 = gate_h.shape
    B3, H = h_prev.shape
    assert B == B2 == B3
    assert threeH == threeH2
    assert threeH % 3 == 0
    assert threeH // 3 == H

    h_new = torch.empty_like(h_prev)

    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_M"]),
        triton.cdiv(H, META["BLOCK_N"]),
    )

    gru_gates_update_kernel[grid](
        gate_x, gate_h,
        h_prev, h_new,
        B, H,
        gate_x.stride(0), gate_x.stride(1),
        gate_h.stride(0), gate_h.stride(1),
        h_prev.stride(0), h_prev.stride(1),
        h_new.stride(0), h_new.stride(1),
        BLOCK_M=32, BLOCK_N=64,
    )
    return h_new


# -----------------------
# High-performance GRU Module
# -----------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias_flag = bias
        self.batch_first = batch_first

        # Parameters follow a simple layout:
        # For each layer l:
        #   W_ih_l: [input_size_l, 3 * hidden_size]
        #   W_hh_l: [hidden_size,   3 * hidden_size]
        #   b_ih_l: [3 * hidden_size]
        #   b_hh_l: [3 * hidden_size]
        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        self.bias_ih = nn.ParameterList()
        self.bias_hh = nn.ParameterList()

        for layer in range(num_layers):
            in_dim = input_size if layer == 0 else hidden_size

            w_ih = nn.Parameter(torch.empty(in_dim, 3 * hidden_size))
            w_hh = nn.Parameter(torch.empty(hidden_size, 3 * hidden_size))
            self.weight_ih.append(w_ih)
            self.weight_hh.append(w_hh)

            if bias:
                b_ih = nn.Parameter(torch.empty(3 * hidden_size))
                b_hh = nn.Parameter(torch.empty(3 * hidden_size))
            else:
                # Zero biases, no gradient (equivalent to bias=False)
                b_ih = nn.Parameter(torch.zeros(3 * hidden_size), requires_grad=False)
                b_hh = nn.Parameter(torch.zeros(3 * hidden_size), requires_grad=False)
            self.bias_ih.append(b_ih)
            self.bias_hh.append(b_hh)

        self.reset_parameters()

    def reset_parameters(self):
        # Simple initialization similar in spirit to PyTorch GRU
        for l in range(self.num_layers):
            w_ih = self.weight_ih[l]
            w_hh = self.weight_hh[l]
            nn.init.xavier_uniform_(w_ih)
            nn.init.orthogonal_(w_hh)
            if self.bias_flag:
                nn.init.zeros_(self.bias_ih[l])
                nn.init.zeros_(self.bias_hh[l])

    def forward(self, x, h0):
        """
        x: (seq_len, batch, input_size) if batch_first=False
           (batch, seq_len, input_size) if batch_first=True
        h0: (num_layers, batch, hidden_size)
        Returns:
          output: (seq_len, batch, hidden_size) or (batch, seq_len, hidden_size)
          h_n: (num_layers, batch, hidden_size)
        """
        if self.batch_first:
            # (B, T, C) -> (T, B, C)
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        device = x.device

        # Ensure h0 on same device and type
        h_prev = h0.to(device=device, dtype=x.dtype).clone()  # [L, B, H]

        outputs_last_layer = []

        for t in range(seq_len):
            x_t = x[t]  # [B, input_size] (for layer 0)
            layer_input = x_t

            for layer in range(self.num_layers):
                h_prev_l = h_prev[layer]  # [B, H]
                in_dim = self.input_size if layer == 0 else self.hidden_size

                # x projection: gate_x = x_t * W_ih + b_ih
                W_ih = self.weight_ih[layer]
                W_hh = self.weight_hh[layer]
                b_ih = self.bias_ih[layer]
                b_hh = self.bias_hh[layer]

                # Ensure inputs are on the same device and type as weights
                layer_input = layer_input.to(device=device, dtype=W_ih.dtype)
                h_prev_l = h_prev_l.to(device=device, dtype=W_hh.dtype)

                gate_x = gru_gemm_bias(layer_input, W_ih, b_ih)   # [B, 3H]
                gate_h = gru_gemm_bias(h_prev_l, W_hh, b_hh)      # [B, 3H]

                h_new_l = gru_gates_update(gate_x, gate_h, h_prev_l)  # [B, H]

                # Update hidden state for this layer
                h_prev[layer] = h_new_l

                # Output of this layer is input to next layer at same time step
                layer_input = h_new_l

            # layer_input is output of last layer at time t
            outputs_last_layer.append(layer_input)

        # Stack outputs over time
        output = torch.stack(outputs_last_layer, dim=0)  # [T, B, H]
        h_n = h_prev  # [L, B, H]

        if self.batch_first:
            output = output.transpose(0, 1)  # (B, T, H)

        return output, h_n
