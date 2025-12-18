import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def lstm_step_kernel(
    x_ptr,                  # (B, I)
    h_prev_ptr,             # (B, H)
    c_prev_ptr,             # (B, H)
    w_ii_ptr, w_if_ptr, w_ig_ptr, w_io_ptr,  # (I, H) each
    w_hi_ptr, w_hf_ptr, w_hg_ptr, w_ho_ptr,  # (H, H) each
    b_i_ptr, b_f_ptr, b_g_ptr, b_o_ptr,      # (H,) each
    h_out_ptr,              # (B, H)
    c_out_ptr,              # (B, H)
    B, I, H,
    stride_x_b, stride_x_i,
    stride_h_b, stride_h_h,
    stride_c_b, stride_c_h,
    stride_wii_k, stride_wii_h,
    stride_wif_k, stride_wif_h,
    stride_wig_k, stride_wig_h,
    stride_wio_k, stride_wio_h,
    stride_whi_k, stride_whi_h,
    stride_whf_k, stride_whf_h,
    stride_whg_k, stride_whg_h,
    stride_who_k, stride_who_h,
    stride_ho_b, stride_ho_h,
    stride_co_b, stride_co_h,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_k = tl.arange(0, BLOCK_K)

    mask_b = offs_b < B
    mask_h = offs_h < H
    mask = mask_b[:, None] & mask_h[None, :]

    acc_i = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    acc_f = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)

    # Input contributions: x @ W_?i , W_?i shape (I, H)
    for k in range(0, I, BLOCK_K):
        k_offsets = k + offs_k
        k_mask = k_offsets < I

        x_ptrs = x_ptr + offs_b[:, None] * stride_x_b + k_offsets[None, :] * stride_x_i
        x_tile = tl.load(x_ptrs, mask=mask_b[:, None] & k_mask[None, :], other=0.0)
        x_tile = x_tile.to(tl.float32)

        wii_ptrs = w_ii_ptr + k_offsets[:, None] * stride_wii_k + offs_h[None, :] * stride_wii_h
        wif_ptrs = w_if_ptr + k_offsets[:, None] * stride_wif_k + offs_h[None, :] * stride_wif_h
        wig_ptrs = w_ig_ptr + k_offsets[:, None] * stride_wig_k + offs_h[None, :] * stride_wig_h
        wio_ptrs = w_io_ptr + k_offsets[:, None] * stride_wio_k + offs_h[None, :] * stride_wio_h

        wii = tl.load(wii_ptrs, mask=k_mask[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        wif = tl.load(wif_ptrs, mask=k_mask[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        wig = tl.load(wig_ptrs, mask=k_mask[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        wio = tl.load(wio_ptrs, mask=k_mask[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

        acc_i += tl.dot(x_tile, wii, allow_tf32=True)
        acc_f += tl.dot(x_tile, wif, allow_tf32=True)
        acc_g += tl.dot(x_tile, wig, allow_tf32=True)
        acc_o += tl.dot(x_tile, wio, allow_tf32=True)

    # Hidden contributions: h_prev @ W_?h , W_?h shape (H, H)
    for k in range(0, H, BLOCK_K):
        k_offsets = k + offs_k
        k_mask = k_offsets < H

        h_prev_ptrs = h_prev_ptr + offs_b[:, None] * stride_h_b + k_offsets[None, :] * stride_h_h
        h_tile = tl.load(h_prev_ptrs, mask=mask_b[:, None] & k_mask[None, :], other=0.0)
        h_tile = h_tile.to(tl.float32)

        whi_ptrs = w_hi_ptr + k_offsets[:, None] * stride_whi_k + offs_h[None, :] * stride_whi_h
        whf_ptrs = w_hf_ptr + k_offsets[:, None] * stride_whf_k + offs_h[None, :] * stride_whf_h
        whg_ptrs = w_hg_ptr + k_offsets[:, None] * stride_whg_k + offs_h[None, :] * stride_whg_h
        who_ptrs = w_ho_ptr + k_offsets[:, None] * stride_who_k + offs_h[None, :] * stride_who_h

        whi = tl.load(whi_ptrs, mask=k_mask[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        whf = tl.load(whf_ptrs, mask=k_mask[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        whg = tl.load(whg_ptrs, mask=k_mask[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        who = tl.load(who_ptrs, mask=k_mask[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

        acc_i += tl.dot(h_tile, whi, allow_tf32=True)
        acc_f += tl.dot(h_tile, whf, allow_tf32=True)
        acc_g += tl.dot(h_tile, whg, allow_tf32=True)
        acc_o += tl.dot(h_tile, who, allow_tf32=True)

    # Add biases
    bi = tl.load(b_i_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    bf = tl.load(b_f_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    bg = tl.load(b_g_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    bo = tl.load(b_o_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    acc_i += bi[None, :]
    acc_f += bf[None, :]
    acc_g += bg[None, :]
    acc_o += bo[None, :]

    # Load previous cell state
    c_prev_ptrs = c_prev_ptr + offs_b[:, None] * stride_c_b + offs_h[None, :] * stride_c_h
    c_prev = tl.load(c_prev_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Activations: sigmoid and tanh via exp
    one = 1.0
    two = 2.0

    i_gate = one / (one + tl.exp(-acc_i))
    f_gate = one / (one + tl.exp(-acc_f))
    o_gate = one / (one + tl.exp(-acc_o))

    g_tmp = two * acc_g
    g_gate = two / (one + tl.exp(-g_tmp)) - one  # tanh(acc_g)

    c_t = f_gate * c_prev + i_gate * g_gate

    ct_tmp = two * c_t
    tanh_ct = two / (one + tl.exp(-ct_tmp)) - one  # tanh(c_t)

    h_t = o_gate * tanh_ct

    # Store outputs
    h_out_ptrs = h_out_ptr + offs_b[:, None] * stride_ho_b + offs_h[None, :] * stride_ho_h
    c_out_ptrs = c_out_ptr + offs_b[:, None] * stride_co_b + offs_h[None, :] * stride_co_h

    tl.store(h_out_ptrs, h_t, mask=mask)
    tl.store(c_out_ptrs, c_t, mask=mask)


def lstm_step_triton(
    x_t: torch.Tensor,
    h_prev: torch.Tensor,
    c_prev: torch.Tensor,
    w_ii: torch.Tensor,
    w_if: torch.Tensor,
    w_ig: torch.Tensor,
    w_io: torch.Tensor,
    w_hi: torch.Tensor,
    w_hf: torch.Tensor,
    w_hg: torch.Tensor,
    w_ho: torch.Tensor,
    b_i: torch.Tensor,
    b_f: torch.Tensor,
    b_g: torch.Tensor,
    b_o: torch.Tensor,
) -> (torch.Tensor, torch.Tensor):
    """
    Single LSTM layer step over one time step, for the whole batch.
    x_t:   (B, I)
    h_prev,c_prev: (B, H)
    weights: (I,H) / (H,H), biases: (H,)
    """
    assert x_t.is_cuda and h_prev.is_cuda and c_prev.is_cuda, "Inputs must be CUDA tensors"
    B, I = x_t.shape
    B_h, H = h_prev.shape
    assert B == B_h, "Batch size mismatch between x_t and h_prev"
    assert c_prev.shape == h_prev.shape, "c_prev must match h_prev shape"

    h_out = torch.empty_like(h_prev)
    c_out = torch.empty_like(c_prev)

    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_B"]),
        triton.cdiv(H, META["BLOCK_H"]),
    )

    lstm_step_kernel[grid](
        x_t,
        h_prev,
        c_prev,
        w_ii,
        w_if,
        w_ig,
        w_io,
        w_hi,
        w_hf,
        w_hg,
        w_ho,
        b_i,
        b_f,
        b_g,
        b_o,
        h_out,
        c_out,
        B,
        I,
        H,
        x_t.stride(0),
        x_t.stride(1),
        h_prev.stride(0),
        h_prev.stride(1),
        c_prev.stride(0),
        c_prev.stride(1),
        w_ii.stride(0),
        w_ii.stride(1),
        w_if.stride(0),
        w_if.stride(1),
        w_ig.stride(0),
        w_ig.stride(1),
        w_io.stride(0),
        w_io.stride(1),
        w_hi.stride(0),
        w_hi.stride(1),
        w_hf.stride(0),
        w_hf.stride(1),
        w_hg.stride(0),
        w_hg.stride(1),
        w_ho.stride(0),
        w_ho.stride(1),
        h_out.stride(0),
        h_out.stride(1),
        c_out.stride(0),
        c_out.stride(1),
        BLOCK_B=32,
        BLOCK_H=64,
        BLOCK_K=32,
    )

    return h_out, c_out


class LSTMLayerTriton(nn.Module):
    """
    One LSTM layer implemented using a Triton kernel for the per-time-step update.
    Weights layout:
      W_ii, W_if, W_ig, W_io: (input_size, hidden_size)
      W_hi, W_hf, W_hg, W_ho: (hidden_size, hidden_size)
      biases: (hidden_size,) each
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input weights
        self.w_ii = nn.Parameter(torch.empty(input_size, hidden_size))
        self.w_if = nn.Parameter(torch.empty(input_size, hidden_size))
        self.w_ig = nn.Parameter(torch.empty(input_size, hidden_size))
        self.w_io = nn.Parameter(torch.empty(input_size, hidden_size))

        # Recurrent weights
        self.w_hi = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hf = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hg = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_ho = nn.Parameter(torch.empty(hidden_size, hidden_size))

        # Biases
        self.b_i = nn.Parameter(torch.empty(hidden_size))
        self.b_f = nn.Parameter(torch.empty(hidden_size))
        self.b_g = nn.Parameter(torch.empty(hidden_size))
        self.b_o = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Simple Xavier initialization for weights, zeros for biases
        for w in [
            self.w_ii,
            self.w_if,
            self.w_ig,
            self.w_io,
            self.w_hi,
            self.w_hf,
            self.w_hg,
            self.w_ho,
        ]:
            nn.init.xavier_uniform_(w)
        for b in [self.b_i, self.b_f, self.b_g, self.b_o]:
            nn.init.zeros_(b)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        return lstm_step_triton(
            x_t,
            h_prev,
            c_prev,
            self.w_ii,
            self.w_if,
            self.w_ig,
            self.w_io,
            self.w_hi,
            self.w_hf,
            self.w_hg,
            self.w_ho,
            self.b_i,
            self.b_f,
            self.b_g,
            self.b_o,
        )


class ModelNew(nn.Module):
    """
    LSTM stack implemented with Triton kernels for the recurrent updates.
    Matches the interface of the original Model but returns only the final cell state c_n
    (state[1] from nn.LSTM), shape (num_layers, batch, hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = float(dropout)

        layers = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            layers.append(LSTMLayerTriton(layer_input_size, hidden_size))
        self.layers = nn.ModuleList(layers)

        # Keep a Linear to mirror the original structure, though its output
        # is not used in the final return value (original Model discards fc output).
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        """
        x:  (batch, seq_len, input_size)
        h0: (num_layers, batch, hidden_size)
        c0: (num_layers, batch, hidden_size)
        Returns:
          c_n: (num_layers, batch, hidden_size)  -- final cell states
        """
        assert x.dim() == 3, "x must be (batch, seq_len, input_size)"
        batch_size, seq_len, _ = x.shape
        assert h0.shape[0] == self.num_layers
        assert c0.shape == h0.shape

        # Ensure contiguous for recurrent states
        h_prev = [h0[layer].contiguous() for layer in range(self.num_layers)]
        c_prev = [c0[layer].contiguous() for layer in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)
            for layer_idx, layer in enumerate(self.layers):
                h_t, c_t = layer(x_t, h_prev[layer_idx], c_prev[layer_idx])

                # Apply dropout between layers (not on last layer), only in training
                if (
                    self.dropout_p > 0.0
                    and layer_idx < self.num_layers - 1
                    and self.training
                ):
                    h_t = nn.functional.dropout(h_t, p=self.dropout_p, training=True)

                h_prev[layer_idx] = h_t
                c_prev[layer_idx] = c_t
                x_t = h_t  # input to next layer

        # Final cell states stacked over layers
        c_n = torch.stack(c_prev, dim=0)  # (num_layers, batch, hidden_size)
        return c_n
