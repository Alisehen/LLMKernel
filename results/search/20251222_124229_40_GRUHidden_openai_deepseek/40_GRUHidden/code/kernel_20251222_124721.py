import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=['B', 'H', 'Kx'],
)
@triton.jit
def gru_cell_kernel(
    x_ptr, h_prev_ptr,
    w_ir_ptr, w_iz_ptr, w_in_ptr,
    w_hr_ptr, w_hz_ptr, w_hn_ptr,
    b_ir_ptr, b_iz_ptr, b_in_ptr,
    b_hr_ptr, b_hz_ptr, b_hn_ptr,
    h_new_ptr,
    B, Kx, H,
    stride_xb, stride_xk,
    stride_hb, stride_hh,
    stride_wirk, stride_wirh,
    stride_wizk, stride_wizh,
    stride_wink, stride_winh,
    stride_whrk, stride_whrh,
    stride_whzk, stride_whzh,
    stride_whnk, stride_whnh,
    stride_hnewb, stride_hnewh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over (batch, hidden)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_b = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_b = offs_b < B
    mask_h = offs_h < H
    mask_bh = mask_b[:, None] & mask_h[None, :]

    # ----- Accumulators -----
    # x * W_ih (input-to-hidden) contributions
    acc_ir = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_iz = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_in = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # h_prev * W_hh (hidden-to-hidden) contributions
    acc_hr = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_hz = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_hn = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ----- Input-to-hidden GEMMs: x @ {W_ir, W_iz, W_in} -----
    k = 0
    while k < Kx:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Kx

        # [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + (offs_b[:, None] * stride_xb) + (offs_k[None, :] * stride_xk)
        x_block = tl.load(x_ptrs, mask=mask_b[:, None] & mask_k[None, :], other=0.0)
        x_block = x_block.to(tl.float32)

        # [BLOCK_K, BLOCK_N] tiles from each weight matrix
        wir_ptrs = w_ir_ptr + (offs_k[:, None] * stride_wirk) + (offs_h[None, :] * stride_wirh)
        wiz_ptrs = w_iz_ptr + (offs_k[:, None] * stride_wizk) + (offs_h[None, :] * stride_wizh)
        win_ptrs = w_in_ptr + (offs_k[:, None] * stride_wink) + (offs_h[None, :] * stride_winh)

        wir = tl.load(wir_ptrs, mask=mask_k[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        wiz = tl.load(wiz_ptrs, mask=mask_k[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        win = tl.load(win_ptrs, mask=mask_k[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

        acc_ir += tl.dot(x_block, wir, allow_tf32=True)
        acc_iz += tl.dot(x_block, wiz, allow_tf32=True)
        acc_in += tl.dot(x_block, win, allow_tf32=True)

        k += BLOCK_K

    # ----- Hidden-to-hidden GEMMs: h_prev @ {W_hr, W_hz, W_hn} -----
    k_h = 0
    while k_h < H:
        offs_kh = k_h + tl.arange(0, BLOCK_K)
        mask_kh = offs_kh < H

        # [BLOCK_M, BLOCK_K]
        h_ptrs = h_prev_ptr + (offs_b[:, None] * stride_hb) + (offs_kh[None, :] * stride_hh)
        h_block = tl.load(h_ptrs, mask=mask_b[:, None] & mask_kh[None, :], other=0.0)
        h_block = h_block.to(tl.float32)

        # [BLOCK_K, BLOCK_N] tiles from each hidden weight matrix
        whr_ptrs = w_hr_ptr + (offs_kh[:, None] * stride_whrk) + (offs_h[None, :] * stride_whrh)
        whz_ptrs = w_hz_ptr + (offs_kh[:, None] * stride_whzk) + (offs_h[None, :] * stride_whzh)
        whn_ptrs = w_hn_ptr + (offs_kh[:, None] * stride_whnk) + (offs_h[None, :] * stride_whnh)

        whr = tl.load(whr_ptrs, mask=mask_kh[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        whz = tl.load(whz_ptrs, mask=mask_kh[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
        whn = tl.load(whn_ptrs, mask=mask_kh[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

        acc_hr += tl.dot(h_block, whr, allow_tf32=True)
        acc_hz += tl.dot(h_block, whz, allow_tf32=True)
        acc_hn += tl.dot(h_block, whn, allow_tf32=True)

        k_h += BLOCK_K

    # ----- Bias add (broadcast over batch) -----
    # All following fused ops use the same offsets (offs_b, offs_h) and mask (mask_bh)
    b_ir = tl.load(b_ir_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    b_iz = tl.load(b_iz_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    b_in = tl.load(b_in_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    b_hr = tl.load(b_hr_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    b_hz = tl.load(b_hz_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    b_hn = tl.load(b_hn_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    gi_r = acc_ir + b_ir[None, :]
    gi_z = acc_iz + b_iz[None, :]
    gi_n = acc_in + b_in[None, :]

    gh_r = acc_hr + b_hr[None, :]
    gh_z = acc_hz + b_hz[None, :]
    gh_n = acc_hn + b_hn[None, :]

    # ----- GRU non-linearities (all on [B, H] tile with shared offsets/mask) -----
    # r = sigmoid(gi_r + gh_r)
    # z = sigmoid(gi_z + gh_z)
    r_pre = gi_r + gh_r
    z_pre = gi_z + gh_z

    # sigmoid(x) = 1 / (1 + exp(-x))
    r = 1.0 / (1.0 + tl.exp(-r_pre))
    z = 1.0 / (1.0 + tl.exp(-z_pre))

    # n_tilde = tanh(gi_n + r * gh_n)
    n_pre = gi_n + r * gh_n
    # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    e2 = tl.exp(2.0 * n_pre)
    n_tilde = (e2 - 1.0) / (e2 + 1.0)

    # Load previous hidden state tile on same offsets/mask
    h_prev_tile_ptrs = h_prev_ptr + (offs_b[:, None] * stride_hb) + (offs_h[None, :] * stride_hh)
    h_prev_tile = tl.load(h_prev_tile_ptrs, mask=mask_bh, other=0.0).to(tl.float32)

    # h_new = (1 - z) * n_tilde + z * h_prev
    h_new = (1.0 - z) * n_tilde + z * h_prev_tile

    # Store result
    h_new_ptrs = h_new_ptr + (offs_b[:, None] * stride_hnewb) + (offs_h[None, :] * stride_hnewh)
    tl.store(h_new_ptrs, h_new.to(h_prev_tile.dtype), mask=mask_bh)


def gru_cell_triton(x, h_prev, w_ih, w_hh, b_ih, b_hh):
    """
    x:      [B, Kx]
    h_prev: [B, H]
    w_ih:   [3H, Kx]
    w_hh:   [3H, H]
    b_ih:   [3H] or None
    b_hh:   [3H] or None
    returns h_new: [B, H]
    """
    assert x.is_cuda and h_prev.is_cuda, "Inputs must be CUDA tensors for Triton kernels"

    B, Kx = x.shape
    B_h, H = h_prev.shape
    assert B == B_h, "Batch dimension mismatch between x and h_prev"

    # Split gate weights: [3H, K] -> three [H, K]
    w_ir, w_iz, w_in = w_ih.split(H, dim=0)
    w_hr, w_hz, w_hn = w_hh.split(H, dim=0)

    # Transpose to [K, H] for better GEMM layout
    w_ir_t = w_ir.t().contiguous()
    w_iz_t = w_iz.t().contiguous()
    w_in_t = w_in.t().contiguous()

    w_hr_t = w_hr.t().contiguous()
    w_hz_t = w_hz.t().contiguous()
    w_hn_t = w_hn.t().contiguous()

    # Biases per gate
    if b_ih is None:
        b_ir = torch.zeros(H, device=x.device, dtype=x.dtype)
        b_iz = torch.zeros(H, device=x.device, dtype=x.dtype)
        b_in = torch.zeros(H, device=x.device, dtype=x.dtype)
    else:
        b_ir, b_iz, b_in = b_ih.split(H, dim=0)

    if b_hh is None:
        b_hr = torch.zeros(H, device=x.device, dtype=x.dtype)
        b_hz = torch.zeros(H, device=x.device, dtype=x.dtype)
        b_hn = torch.zeros(H, device=x.device, dtype=x.dtype)
    else:
        b_hr, b_hz, b_hn = b_hh.split(H, dim=0)

    x_c = x.contiguous()
    h_prev_c = h_prev.contiguous()

    h_new = torch.empty_like(h_prev_c)

    def grid(META):
        return (
            triton.cdiv(B, META['BLOCK_M']),
            triton.cdiv(H, META['BLOCK_N']),
        )

    gru_cell_kernel[grid](
        x_c, h_prev_c,
        w_ir_t, w_iz_t, w_in_t,
        w_hr_t, w_hz_t, w_hn_t,
        b_ir, b_iz, b_in,
        b_hr, b_hz, b_hn,
        h_new,
        B, Kx, H,
        x_c.stride(0), x_c.stride(1),
        h_prev_c.stride(0), h_prev_c.stride(1),
        w_ir_t.stride(0), w_ir_t.stride(1),
        w_iz_t.stride(0), w_iz_t.stride(1),
        w_in_t.stride(0), w_in_t.stride(1),
        w_hr_t.stride(0), w_hr_t.stride(1),
        w_hz_t.stride(0), w_hz_t.stride(1),
        w_hn_t.stride(0), w_hn_t.stride(1),
        h_new.stride(0), h_new.stride(1),
    )

    return h_new


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        self.weight_ih_l = nn.ParameterList()
        self.weight_hh_l = nn.ParameterList()
        if bias:
            self.bias_ih_l = nn.ParameterList()
            self.bias_hh_l = nn.ParameterList()
        else:
            self.bias_ih_l = None
            self.bias_hh_l = None

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size

            w_ih = nn.Parameter(torch.empty(3 * hidden_size, layer_input_size))
            w_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            nn.init.xavier_uniform_(w_ih)
            nn.init.orthogonal_(w_hh)

            self.weight_ih_l.append(w_ih)
            self.weight_hh_l.append(w_hh)

            if bias:
                b_ih = nn.Parameter(torch.zeros(3 * hidden_size))
                b_hh = nn.Parameter(torch.zeros(3 * hidden_size))
                self.bias_ih_l.append(b_ih)
                self.bias_hh_l.append(b_hh)

    def forward(self, x, h0):
        # x: (T, B, input) if not batch_first else (B, T, input)
        # h0: (num_layers, B, H)
        if self.batch_first:
            x_seq = x.transpose(0, 1).contiguous()
        else:
            x_seq = x.contiguous()

        T, B, _ = x_seq.shape
        assert h0.shape[0] == self.num_layers
        assert h0.shape[1] == B
        assert h0.shape[2] == self.hidden_size

        layer_input = x_seq
        h_n_list = []

        for layer in range(self.num_layers):
            h_prev = h0[layer].contiguous()
            w_ih = self.weight_ih_l[layer]
            w_hh = self.weight_hh_l[layer]
            b_ih = self.bias_ih_l[layer] if self.bias else None
            b_hh = self.bias_hh_l[layer] if self.bias else None

            # output sequence for this layer
            layer_output = torch.empty(
                T, B, self.hidden_size,
                device=x_seq.device,
                dtype=x_seq.dtype,
            )

            for t in range(T):
                x_t = layer_input[t]
                h_prev = gru_cell_triton(x_t, h_prev, w_ih, w_hh, b_ih, b_hh)
                layer_output[t] = h_prev

            h_n_list.append(h_prev)
            layer_input = layer_output

        h_n = torch.stack(h_n_list, dim=0)
        return h_n
