import torch, torch.nn as nn, triton, triton.language as tl
import math


# =========================================
# Triton kernels
# =========================================

@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Computes C = A @ B + bias, where:
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

    # K loop
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

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

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_out)


@triton.jit
def gru_cell_kernel(
    gates_x_ptr, gates_h_ptr, h_prev_ptr, h_new_ptr,
    B, H,
    stride_gxb, stride_gxg,
    stride_ghb, stride_ghg,
    stride_hb, stride_hh,
    BLOCK_H: tl.constexpr,
):
    """
    One GRU cell step for a full batch:
      inputs:
        gates_x: [B, 3H] = x @ W_ih^T + b_ih
        gates_h: [B, 3H] = h_prev @ W_hh^T + b_hh
        h_prev:  [B, H]
      outputs:
        h_new:   [B, H]

    PyTorch GRUCell equations:

      gi = x @ W_ih^T + b_ih
      gh = h_prev @ W_hh^T + b_hh
      i_r, i_z, i_n = gi.chunk(3, 1)
      h_r, h_z, h_n = gh.chunk(3, 1)

      r = sigmoid(i_r + h_r)
      z = sigmoid(i_z + h_z)
      n = tanh(i_n + r * h_n)
      h_new = n + z * (h_prev - n)
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # Base pointers for this batch element
    gx_row = gates_x_ptr + pid_b * stride_gxb
    gh_row = gates_h_ptr + pid_b * stride_ghb
    h_prev_row = h_prev_ptr + pid_b * stride_hb
    h_new_row = h_new_ptr + pid_b * stride_hb

    # Offsets for r, z, n segments
    # indices in gate dim: [0:H) -> r, [H:2H) -> z, [2H:3H) -> n
    offs_r = offs_h
    offs_z = offs_h + H
    offs_n = offs_h + 2 * H

    # Load gated inputs
    gx_r = tl.load(gx_row + offs_r * stride_gxg, mask=mask_h, other=0.0)
    gx_z = tl.load(gx_row + offs_z * stride_gxg, mask=mask_h, other=0.0)
    gx_n = tl.load(gx_row + offs_n * stride_gxg, mask=mask_h, other=0.0)

    gh_r = tl.load(gh_row + offs_r * stride_ghg, mask=mask_h, other=0.0)
    gh_z = tl.load(gh_row + offs_z * stride_ghg, mask=mask_h, other=0.0)
    gh_n = tl.load(gh_row + offs_n * stride_ghg, mask=mask_h, other=0.0)

    h_prev = tl.load(h_prev_row + offs_h * stride_hh, mask=mask_h, other=0.0)

    # i_r + h_r, i_z + h_z, i_n
    ir = gx_r + gh_r
    iz = gx_z + gh_z
    in_ = gx_n  # i_n part

    # sigmoid(x) = 1 / (1 + exp(-x))
    r = 1.0 / (1.0 + tl.exp(-ir))
    z = 1.0 / (1.0 + tl.exp(-iz))

    # n = tanh(i_n + r * h_n)
    n_input = in_ + r * gh_n
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    exp_2x = tl.exp(2.0 * n_input)
    n = (exp_2x - 1.0) / (exp_2x + 1.0)

    # h_new = n + z * (h_prev - n)
    h_new = n + z * (h_prev - n)

    tl.store(h_new_row + offs_h * stride_hh, h_new, mask=mask_h)


# =========================================
# Python wrappers around kernels
# =========================================

def triton_linear_bias(a: torch.Tensor, w_t: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Compute a @ w^T + bias using Triton, where w_t is w.T:
      a:   [M, K]
      w_t: [K, N]
      bias:[N]
      out: [M, N]
    """
    assert a.is_cuda and w_t.is_cuda and bias.is_cuda
    M, K = a.shape
    K_w, N = w_t.shape
    assert K == K_w
    assert bias.numel() == N

    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    matmul_bias_kernel[grid](
        a, w_t, bias, out,
        M, N, K,
        a.stride(0), a.stride(1),
        w_t.stride(0), w_t.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out


def triton_gru_cell_step(
    gates_x: torch.Tensor,
    h_prev: torch.Tensor,
    w_hh_t: torch.Tensor,
    bias_hh: torch.Tensor,
) -> torch.Tensor:
    """
    One GRU cell step for full batch using Triton kernels:
      gates_x: [B, 3H]  (precomputed input contribution)
      h_prev:  [B, H]
      w_hh_t:  [H, 3H]  (transpose of weight_hh [3H, H])
      bias_hh: [3H]
    Returns:
      h_new:   [B, H]
    """
    B, threeH = gates_x.shape
    H = h_prev.shape[1]
    assert threeH == 3 * H
    assert w_hh_t.shape == (H, threeH)
    assert bias_hh.numel() == threeH

    # Compute gates_h = h_prev @ w_hh^T + b_hh
    gates_h = triton_linear_bias(h_prev, w_hh_t, bias_hh)

    # Now run GRU cell kernel
    h_new = torch.empty_like(h_prev)

    BLOCK_H = 128
    grid = lambda META: (
        B,
        triton.cdiv(H, META["BLOCK_H"]),
    )

    gru_cell_kernel[grid](
        gates_x, gates_h, h_prev, h_new,
        B, H,
        gates_x.stride(0), gates_x.stride(1),
        gates_h.stride(0), gates_h.stride(1),
        h_prev.stride(0), h_prev.stride(1),
        BLOCK_H=BLOCK_H,
    )

    return h_new


# =========================================
# Triton-based GRU module
# =========================================

class TritonGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # Match nn.GRU parameter naming/layout for state_dict compatibility
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size

            weight_ih = nn.Parameter(torch.empty(3 * hidden_size, layer_input_size))
            weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            self.register_parameter(f"weight_ih_l{layer}", weight_ih)
            self.register_parameter(f"weight_hh_l{layer}", weight_hh)

            if bias:
                bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
                bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
                self.register_parameter(f"bias_ih_l{layer}", bias_ih)
                self.register_parameter(f"bias_hh_l{layer}", bias_hh)
            else:
                # Still register zero biases to simplify kernel calls
                bias_ih = nn.Parameter(torch.zeros(3 * hidden_size), requires_grad=False)
                bias_hh = nn.Parameter(torch.zeros(3 * hidden_size), requires_grad=False)
                self.register_parameter(f"bias_ih_l{layer}", bias_ih)
                self.register_parameter(f"bias_hh_l{layer}", bias_hh)

        self.reset_parameters()

    def reset_parameters(self):
        # Similar to nn.GRU default initialization
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            with torch.no_grad():
                weight.uniform_(-stdv, stdv)

    def forward(self, x, h0=None):
        """
        x:   [seq_len, batch, input_size] if not batch_first
             [batch, seq_len, input_size] if batch_first
        h0:  [num_layers, batch, hidden_size]
        Returns:
          output: [seq_len, batch, hidden_size] (or batch_first)
          h_n:    [num_layers, batch, hidden_size]
        """
        # Arrange input as [seq_len, batch, feature]
        if self.batch_first:
            x = x.transpose(0, 1)  # [T, B, I]
        T, B, _ = x.shape
        device = x.device

        if h0 is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device, dtype=x.dtype)

        prev_layer_output = x
        h_n_list = []

        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size

            weight_ih = getattr(self, f"weight_ih_l{layer}")
            weight_hh = getattr(self, f"weight_hh_l{layer}")
            bias_ih = getattr(self, f"bias_ih_l{layer}")
            bias_hh = getattr(self, f"bias_hh_l{layer}")

            # Precompute W_ih^T, W_hh^T once per forward
            w_ih_t = weight_ih.transpose(0, 1).contiguous().to(device)
            w_hh_t = weight_hh.transpose(0, 1).contiguous().to(device)

            # Precompute input projection for all time steps: [T, B, 3H]
            inp = prev_layer_output.reshape(T * B, layer_input_size)
            gates_x_flat = triton_linear_bias(inp, w_ih_t, bias_ih.to(device))
            gates_x = gates_x_flat.reshape(T, B, 3 * self.hidden_size)

            # Recurrent over time
            h_t = h0[layer]  # [B, H]
            layer_outputs = []

            for t in range(T):
                gates_x_t = gates_x[t]  # [B, 3H]
                h_t = triton_gru_cell_step(
                    gates_x_t,
                    h_t,
                    w_hh_t,
                    bias_hh.to(device),
                )
                layer_outputs.append(h_t)

            # Stack outputs: [T, B, H]
            prev_layer_output = torch.stack(layer_outputs, dim=0)
            h_n_list.append(h_t)

        output = prev_layer_output  # [T, B, H]
        h_n = torch.stack(h_n_list, dim=0)  # [num_layers, B, H]

        if self.batch_first:
            output = output.transpose(0, 1)  # [B, T, H]

        return output, h_n


# =========================================
# Top-level model using TritonGRU
# =========================================

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.gru = TritonGRU(input_size, hidden_size, num_layers, bias, batch_first, )

    def forward(self, x, h0):
        output, h_n = self.gru(x, h0)
        # Match original Model: only return output
        return output
