import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---- Elementwise activations implemented in Triton ----

@triton.jit
def triton_sigmoid(x):
    # Stable sigmoid: 1 / (1 + exp(-x))
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def triton_tanh(x):
    # Numerically stable tanh implementation
    # For x >= 0: tanh(x) = (1 - e^{-2x}) / (1 + e^{-2x})
    # For x < 0 : tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    two = 2.0
    x2 = two * x

    e_neg = tl.exp(-x2)
    tanh_pos = (1.0 - e_neg) / (1.0 + e_neg)  # x >= 0

    e_pos = tl.exp(x2)
    tanh_neg = (e_pos - 1.0) / (e_pos + 1.0)  # x < 0

    mask_pos = x >= 0
    return tl.where(mask_pos, tanh_pos, tanh_neg)


# ---- Matmul + bias kernel ----

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
    Compute C = A @ B + bias
    A: (M, K)
    B: (K, N)
    bias: (N,)
    C: (M, N)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + offs_k

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak
        b_ptrs = b_ptr + k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_ids[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_ids[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b, allow_tf32=True)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    bias = bias.to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def matmul_bias(a: torch.Tensor, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for matmul_bias_kernel.
    a: (M, K)
    w: (K, N)
    bias: (N,)
    returns c: (M, N)
    """
    assert a.is_cuda and w.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    assert a.dtype == torch.float32 and w.dtype == torch.float32 and bias.dtype == torch.float32

    a_contig = a.contiguous()
    w_contig = w.contiguous()
    bias_contig = bias.contiguous()

    M, K = a_contig.shape
    Kw, N = w_contig.shape
    assert Kw == K, "Incompatible matmul dimensions"
    assert bias_contig.numel() == N

    c = torch.empty((M, N), device=a_contig.device, dtype=torch.float32)

    grid = lambda META: (
        max(1, triton.cdiv(M, META["BLOCK_M"])),
        max(1, triton.cdiv(N, META["BLOCK_N"])),
    )

    matmul_bias_kernel[grid](
        a_contig, w_contig, bias_contig, c,
        M, N, K,
        a_contig.stride(0), a_contig.stride(1),
        w_contig.stride(0), w_contig.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        num_warps=4,
    )
    return c


# ---- Persistent GRU layer kernel ----

@triton.jit
def gru_persistent_layer_kernel(
    gates_x_ptr,      # (T, B, 3H)
    h_state_ptr,      # (B, H)  in-place state buffer
    w_r_ptr, w_z_ptr, w_n_ptr,   # (H, H) each
    b_r_ptr, b_z_ptr, b_n_ptr,   # (H,) each
    y_out_ptr,        # (T, B, H)
    h_last_ptr,       # (B, H)
    T, B, H,
    stride_gt, stride_gb, stride_gh,
    stride_hb, stride_hh,
    stride_wk, stride_wn,
    stride_yt, stride_yb, stride_yh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Persistent GRU kernel for a single layer & direction.

    gates_x: (T, B, 3H)  -- precomputed input-side gates + bias (x @ W_ih + b_ih)
    h_state: (B, H)      -- initial hidden state, updated in-place across time
    w_r, w_z, w_n: (H, H) -- recurrent weights
    b_r, b_z, b_n: (H,)   -- recurrent biases
    y_out: (T, B, H)     -- full output sequence
    h_last: (B, H)       -- final hidden state after T steps
    """
    pid_m = tl.program_id(0)  # batch tiles
    pid_n = tl.program_id(1)  # hidden tiles

    offs_b = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_b = offs_b < B
    mask_h = offs_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    # Load recurrent biases for this hidden tile once
    b_r = tl.load(b_r_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    b_z = tl.load(b_z_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    b_n = tl.load(b_n_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    # Time loop kept inside the kernel (persistent RNN)
    for t in range(0, T):
        # Recurrent matmuls: h_{t-1} @ w_{r,z,n}
        acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_n = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, H, BLOCK_K):
            k_ids = k0 + offs_k

            # Load a tile of previous hidden state along K dimension
            a_ptrs = h_state_ptr + offs_b[:, None] * stride_hb + k_ids[None, :] * stride_hh
            a = tl.load(
                a_ptrs,
                mask=(offs_b[:, None] < B) & (k_ids[None, :] < H),
                other=0.0,
            )
            a = a.to(tl.float32)

            # Load corresponding recurrent weights
            w_r_ptrs = w_r_ptr + k_ids[:, None] * stride_wk + offs_h[None, :] * stride_wn
            w_z_ptrs = w_z_ptr + k_ids[:, None] * stride_wk + offs_h[None, :] * stride_wn
            w_n_ptrs = w_n_ptr + k_ids[:, None] * stride_wk + offs_h[None, :] * stride_wn

            mask_kw = (k_ids[:, None] < H) & (offs_h[None, :] < H)
            w_r = tl.load(w_r_ptrs, mask=mask_kw, other=0.0).to(tl.float32)
            w_z = tl.load(w_z_ptrs, mask=mask_kw, other=0.0).to(tl.float32)
            w_n = tl.load(w_n_ptrs, mask=mask_kw, other=0.0).to(tl.float32)

            acc_r += tl.dot(a, w_r, allow_tf32=True)
            acc_z += tl.dot(a, w_z, allow_tf32=True)
            acc_n += tl.dot(a, w_n, allow_tf32=True)

        # Load previous hidden state tile (for GRU update)
        h_prev_tile = tl.load(
            h_state_ptr + offs_b[:, None] * stride_hb + offs_h[None, :] * stride_hh,
            mask=mask_bh,
            other=0.0,
        ).to(tl.float32)

        # Input-side gate pre-activations from precomputed gates_x
        base_g_t = gates_x_ptr + t * stride_gt

        # r gate: slice [:, :, 0:H]
        ig_r = tl.load(
            base_g_t + offs_b[:, None] * stride_gb + offs_h[None, :] * stride_gh,
            mask=mask_bh,
            other=0.0,
        ).to(tl.float32)

        # z gate: slice [:, :, H:2H]
        ig_z = tl.load(
            base_g_t + offs_b[:, None] * stride_gb + (offs_h[None, :] + H) * stride_gh,
            mask=mask_bh,
            other=0.0,
        ).to(tl.float32)

        # n gate (candidate): slice [:, :, 2H:3H]
        ig_n = tl.load(
            base_g_t + offs_b[:, None] * stride_gb + (offs_h[None, :] + 2 * H) * stride_gh,
            mask=mask_bh,
            other=0.0,
        ).to(tl.float32)

        # Gate pre-activations
        pre_r = ig_r + acc_r + b_r[None, :]
        pre_z = ig_z + acc_z + b_z[None, :]

        # Sigmoid gates
        r = triton_sigmoid(pre_r)
        z = triton_sigmoid(pre_z)

        # Candidate state
        h_n_lin = acc_n + b_n[None, :]
        pre_n = ig_n + r * h_n_lin
        n_tilde = triton_tanh(pre_n)

        # GRU update: h_t = (1 - z) * n_tilde + z * h_{t-1}
        h_new = (1.0 - z) * n_tilde + z * h_prev_tile

        # Store output for this timestep
        y_ptrs = y_out_ptr + t * stride_yt + offs_b[:, None] * stride_yb + offs_h[None, :] * stride_yh
        tl.store(y_ptrs, h_new, mask=mask_bh)

        # Update hidden state buffer for next timestep
        tl.store(
            h_state_ptr + offs_b[:, None] * stride_hb + offs_h[None, :] * stride_hh,
            h_new,
            mask=mask_bh,
        )

    # After final timestep, write last hidden state to h_last
    h_last_tile = tl.load(
        h_state_ptr + offs_b[:, None] * stride_hb + offs_h[None, :] * stride_hh,
        mask=mask_bh,
        other=0.0,
    )
    tl.store(
        h_last_ptr + offs_b[:, None] * stride_hb + offs_h[None, :] * stride_hh,
        h_last_tile,
        mask=mask_bh,
    )


def gru_persistent_layer_triton(
    gates_x: torch.Tensor,  # (T, B, 3H)
    h0: torch.Tensor,       # (B, H)
    w_hh: torch.Tensor,     # (3, H, H)
    b_hh: torch.Tensor,     # (3, H)
) -> (torch.Tensor, torch.Tensor):
    """
    One full GRU layer-direction with a persistent Triton kernel.

    gates_x: (T, B, 3H)  precomputed input-side gates + bias
    h0:      (B, H)      initial hidden state
    w_hh:    (3, H, H)   recurrent weights
    b_hh:    (3, H)      recurrent biases

    Returns:
      y_out:   (T, B, H)  full output sequence for this layer & direction
      h_last:  (B, H)     final hidden state
    """
    assert gates_x.is_cuda and h0.is_cuda and w_hh.is_cuda and b_hh.is_cuda
    assert gates_x.dtype == torch.float32
    assert h0.dtype == torch.float32
    assert w_hh.dtype == torch.float32 and b_hh.dtype == torch.float32

    T, B, threeH = gates_x.shape
    H = h0.shape[1]
    assert threeH == 3 * H
    assert w_hh.shape == (3, H, H)
    assert b_hh.shape == (3, H)

    gates_x_c = gates_x.contiguous()
    h_state = h0.contiguous()

    w_r = w_hh[0].contiguous()
    w_z = w_hh[1].contiguous()
    w_n = w_hh[2].contiguous()

    b_r = b_hh[0].contiguous()
    b_z = b_hh[1].contiguous()
    b_n = b_hh[2].contiguous()

    y_out = torch.empty((T, B, H), device=h0.device, dtype=torch.float32)
    h_last = torch.empty_like(h0, dtype=torch.float32)

    grid = lambda META: (
        max(1, triton.cdiv(B, META["BLOCK_M"])),
        max(1, triton.cdiv(H, META["BLOCK_N"])),
    )

    gru_persistent_layer_kernel[grid](
        gates_x_c,
        h_state,
        w_r, w_z, w_n,
        b_r, b_z, b_n,
        y_out,
        h_last,
        T, B, H,
        gates_x_c.stride(0), gates_x_c.stride(1), gates_x_c.stride(2),
        h_state.stride(0), h_state.stride(1),
        w_r.stride(0), w_r.stride(1),
        y_out.stride(0), y_out.stride(1), y_out.stride(2),
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
        num_warps=4,
    )

    return y_out, h_last


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        self.bias_ih = nn.ParameterList()
        self.bias_hh = nn.ParameterList()

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            for _direction in range(self.num_directions):
                # weight_ih: (3, layer_input_size, hidden_size)
                w_ih = nn.Parameter(torch.randn(3, layer_input_size, hidden_size))
                # weight_hh: (3, hidden_size, hidden_size)
                w_hh = nn.Parameter(torch.randn(3, hidden_size, hidden_size))
                self.weight_ih.append(w_ih)
                self.weight_hh.append(w_hh)
                if bias:
                    b_ih = nn.Parameter(torch.randn(3, hidden_size))
                    b_hh = nn.Parameter(torch.randn(3, hidden_size))
                else:
                    b_ih = nn.Parameter(torch.zeros(3, hidden_size), requires_grad=False)
                    b_hh = nn.Parameter(torch.zeros(3, hidden_size), requires_grad=False)
                self.bias_ih.append(b_ih)
                self.bias_hh.append(b_hh)

    def forward(self, x, h0=None):
        """
        x: (seq_len, batch, input_size) if batch_first=False
           (batch, seq_len, input_size) if batch_first=True
        h0: (num_layers * num_directions, batch, hidden_size)
        returns: output, h_n
          output: (seq_len, batch, num_directions * hidden_size) if batch_first=False
                  (batch, seq_len, num_directions * hidden_size) if batch_first=True
          h_n: (num_layers * num_directions, batch, hidden_size)
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # (T, B, C)

        seq_len, batch_size, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=device,
                dtype=dtype,
            )

        x_seq = x
        final_h_list = []

        for layer in range(self.num_layers):
            layer_input_size = x_seq.shape[2]
            layer_out_fwd = torch.empty(
                seq_len, batch_size, self.hidden_size,
                device=device, dtype=dtype,
            )
            if self.num_directions == 2:
                layer_out_bwd = torch.empty_like(layer_out_fwd)

            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction

                w_ih = self.weight_ih[idx]
                w_hh = self.weight_hh[idx]
                b_ih = self.bias_ih[idx]
                b_hh = self.bias_hh[idx]

                # Select sequence order for this direction
                if direction == 0:
                    seq = x_seq  # forward
                else:
                    seq = torch.flip(x_seq, dims=[0])  # backward: reversed time

                # Flatten sequence: (T*B, layer_input_size)
                seq_flat = seq.reshape(seq_len * batch_size, layer_input_size)

                # Concatenate input weights and biases to compute all 3 gates in one GEMM
                # w_ih: (3, layer_input_size, H) -> cat along output dim -> (layer_input_size, 3H)
                w_ih_cat = torch.cat(
                    [w_ih[0], w_ih[1], w_ih[2]],
                    dim=1,
                )  # (layer_input_size, 3H)
                b_ih_cat = torch.cat(
                    [b_ih[0], b_ih[1], b_ih[2]],
                    dim=0,
                )  # (3H,)

                # Precompute input-side gates for all timesteps: (T*B, 3H)
                gates_x_flat = matmul_bias(
                    seq_flat.to(torch.float32),
                    w_ih_cat.to(torch.float32),
                    b_ih_cat.to(torch.float32),
                )
                # Reshape to (T, B, 3H)
                gates_x = gates_x_flat.view(seq_len, batch_size, 3 * self.hidden_size)

                # Initial hidden state for this layer & direction
                h_prev = h0[idx].to(torch.float32)

                # Persistent GRU kernel over time
                y_dir, h_last = gru_persistent_layer_triton(
                    gates_x,
                    h_prev,
                    w_hh.to(torch.float32),
                    b_hh.to(torch.float32),
                )

                # Map outputs back to original time order
                if direction == 0:
                    # Forward direction: already in correct order
                    layer_out_fwd.copy_(y_dir.to(dtype))
                else:
                    # Backward direction: reverse back to original time indices
                    layer_out_bwd.copy_(torch.flip(y_dir, dims=[0]).to(dtype))

                final_h_list.append(h_last.to(dtype))

            # Prepare input for next layer
            if self.num_directions == 1:
                x_seq = layer_out_fwd
            else:
                x_seq = torch.cat([layer_out_fwd, layer_out_bwd], dim=2)

        output = x_seq
        h_n = torch.stack(final_h_list, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n
