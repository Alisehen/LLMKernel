import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------
# Helper device functions
# -----------------------
@triton.jit
def _triton_sigmoid(x):
    # Numerically-stable sigmoid
    # sigmoid(x) = 1 / (1 + exp(-x))
    # No branching for simplicity & speed; relies on hardware exp
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _triton_tanh(x):
    # tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    e = tl.exp(2.0 * x)
    return (e - 1.0) / (e + 1.0)


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

    Grid:
      pid_m in [0, ceil_div(M, BLOCK_M))
      pid_n in [0, ceil_div(N, BLOCK_N))

    All fused ops (matmul output + bias add) share the same (offs_m, offs_n, mask_out).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Output tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # Mask for output tile; used by all fused elementwise ops
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Pointers for first K-tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32 for better precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    # We iterate in blocks of BLOCK_K. The pointer math handles the base offset; the
    # mask on offs_k ensures we don't read past K on the last iteration.
    k = 0
    while k < K:
        k_remaining = K - k

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matmul accumulate; allow TF32 on supported GPUs (e.g., 4090) for speed
        acc += tl.dot(a, b, allow_tf32=True)

        # Move to next K tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Fused bias add: bias[N] broadcast along M
    # Use the SAME offs_n & a mask derived from mask_out (only along N dimension).
    bias_mask = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
    acc += bias[None, :]

    # Write result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
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

    # 2D grid over output [B, N]
    def grid(meta):
        return (
            triton.cdiv(B, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    # Tuned for 4090 / Ada: 64x64x32 works well for many GEMM sizes
    gru_gemm_bias_kernel[grid](
        x, weight, bias, out,
        B, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4,
        num_stages=3,
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
    gate_x: x_t * W_ih + b_ih      -> [B, 3H]
    gate_h: h_{t-1} * W_hh + b_hh  -> [B, 3H]

    PyTorch GRU formulation ("reset-before"):
        i_r, i_z, i_n = gate_x.chunk(3, dim=1)
        h_r, h_z, h_n = gate_h.chunk(3, dim=1)

        r = sigmoid(i_r + h_r)
        z = sigmoid(i_z + h_z)
        n = tanh(i_n + r * h_n)

        h_t = (1 - z) * n + z * h_{t-1}

    Grid:
      pid_b in [0, ceil_div(B, BLOCK_M))
      pid_h in [0, ceil_div(H, BLOCK_N))

    All fused ops share the SAME (offs_b, offs_h, mask) tuple.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_b = pid_b * BLOCK_M + tl.arange(0, BLOCK_M)   # batch indices
    offs_h = pid_h * BLOCK_N + tl.arange(0, BLOCK_N)   # hidden indices

    # Common mask for all loads/stores in this fused kernel
    mask = (offs_b[:, None] < B) & (offs_h[None, :] < H)

    # Base offsets in each tensor (index space), derived from same (offs_b, offs_h)
    # gate_x / gate_h: [B, 3H] laid out as contiguous along second dim
    gx_base = offs_b[:, None] * stride_gx0 + offs_h[None, :] * stride_gx1
    gh_base = offs_b[:, None] * stride_gh0 + offs_h[None, :] * stride_gh1

    # Pointers for each gate slice (reset r, update z, new n), same index (b, h)
    gx_r_ptrs = gate_x_ptr + gx_base
    gx_z_ptrs = gate_x_ptr + gx_base + H * stride_gx1
    gx_n_ptrs = gate_x_ptr + gx_base + 2 * H * stride_gx1

    gh_r_ptrs = gate_h_ptr + gh_base
    gh_z_ptrs = gate_h_ptr + gh_base + H * stride_gh1
    gh_n_ptrs = gate_h_ptr + gh_base + 2 * H * stride_gh1

    # Load gates
    i_r = tl.load(gx_r_ptrs, mask=mask, other=0.0)
    i_z = tl.load(gx_z_ptrs, mask=mask, other=0.0)
    i_n = tl.load(gx_n_ptrs, mask=mask, other=0.0)

    h_r = tl.load(gh_r_ptrs, mask=mask, other=0.0)
    h_z = tl.load(gh_z_ptrs, mask=mask, other=0.0)
    h_n = tl.load(gh_n_ptrs, mask=mask, other=0.0)

    # Previous hidden
    hp_base = offs_b[:, None] * stride_hp0 + offs_h[None, :] * stride_hp1
    h_prev_ptrs = h_prev_ptr + hp_base
    h_prev = tl.load(h_prev_ptrs, mask=mask, other=0.0)

    # r = sigmoid(i_r + h_r)
    pre_r = i_r + h_r
    r = _triton_sigmoid(pre_r)

    # z = sigmoid(i_z + h_z)
    pre_z = i_z + h_z
    z = _triton_sigmoid(pre_z)

    # n = tanh(i_n + r * h_n)
    pre_n = i_n + r * h_n
    n = _triton_tanh(pre_n)

    # h_t = (1 - z) * n + z * h_prev
    h_t = (1.0 - z) * n + z * h_prev

    hn_base = offs_b[:, None] * stride_hn0 + offs_h[None, :] * stride_hn1
    h_new_ptrs = h_new_ptr + hn_base
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

    def grid(meta):
        return (
            triton.cdiv(B, meta["BLOCK_M"]),
            triton.cdiv(H, meta["BLOCK_N"]),
        )

    # 2D grid over hidden state [B, H]
    gru_gates_update_kernel[grid](
        gate_x, gate_h,
        h_prev, h_new,
        B, H,
        gate_x.stride(0), gate_x.stride(1),
        gate_h.stride(0), gate_h.stride(1),
        h_prev.stride(0), h_prev.stride(1),
        h_new.stride(0), h_new.stride(1),
        BLOCK_M=32, BLOCK_N=64,
        num_warps=4,
        num_stages=2,
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

        # Parameters layout:
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

        # Ensure everything is on the same device/dtype as parameters once,
        # to avoid per-time-step casts.
        device = self.weight_ih[0].device
        dtype = self.weight_ih[0].dtype

        x = x.to(device=device, dtype=dtype)
        h_prev = h0.to(device=device, dtype=dtype).clone()  # [L, B, H]

        outputs_last_layer = []

        for t in range(seq_len):
            x_t = x[t]  # [B, input_size] (for layer 0)
            layer_input = x_t

            for layer in range(self.num_layers):
                h_prev_l = h_prev[layer]  # [B, H]
                in_dim = self.input_size if layer == 0 else self.hidden_size
                assert layer_input.shape[1] == in_dim

                W_ih = self.weight_ih[layer]
                W_hh = self.weight_hh[layer]
                b_ih = self.bias_ih[layer]
                b_hh = self.bias_hh[layer]

                # x projection: gate_x = x_t * W_ih + b_ih
                gate_x = gru_gemm_bias(layer_input, W_ih, b_ih)   # [B, 3H]

                # h projection: gate_h = h_prev * W_hh + b_hh
                gate_h = gru_gemm_bias(h_prev_l, W_hh, b_hh)      # [B, 3H]

                # Fused gate computation and state update
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
