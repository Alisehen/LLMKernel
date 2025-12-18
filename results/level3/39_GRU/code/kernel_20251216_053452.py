# <complete ModelNew code with optimized Triton kernels>

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gru_step_kernel(
    x_gates_ptr,  # [B, 3H], gate order [r, z, n]
    h_gates_ptr,  # [B, 3H], gate order [r, z, n]
    h_prev_ptr,   # [B, H]
    h_new_ptr,    # [B, H]
    B, H,
    stride_xb, stride_xh,
    stride_hgb, stride_hgh,
    stride_hpb, stride_hph,
    stride_hnb, stride_hnh,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h_block = tl.program_id(1)

    offs_b = pid_b
    offs_h = pid_h_block * BLOCK_H + tl.arange(0, BLOCK_H)

    mask = (offs_b < B) & (offs_h < H)

    # x_gates / h_gates layout: [B, 3H]
    # [0:H]   = r (reset)
    # [H:2H]  = z (update)
    # [2H:3H] = n (new / candidate)

    # r gate (reset)
    x_r_ptrs = x_gates_ptr + offs_b * stride_xb + offs_h * stride_xh
    h_r_ptrs = h_gates_ptr + offs_b * stride_hgb + offs_h * stride_hgh
    x_r = tl.load(x_r_ptrs, mask=mask, other=0.0)
    h_r = tl.load(h_r_ptrs, mask=mask, other=0.0)
    r_pre = x_r + h_r
    r = 1.0 / (1.0 + tl.exp(-r_pre))  # sigmoid

    # z gate (update)
    x_z_ptrs = x_gates_ptr + offs_b * stride_xb + (offs_h + H) * stride_xh
    h_z_ptrs = h_gates_ptr + offs_b * stride_hgb + (offs_h + H) * stride_hgh
    x_z = tl.load(x_z_ptrs, mask=mask, other=0.0)
    h_z = tl.load(h_z_ptrs, mask=mask, other=0.0)
    z_pre = x_z + h_z
    z = 1.0 / (1.0 + tl.exp(-z_pre))  # sigmoid

    # n gate (candidate)
    x_n_ptrs = x_gates_ptr + offs_b * stride_xb + (offs_h + 2 * H) * stride_xh
    h_n_ptrs = h_gates_ptr + offs_b * stride_hgb + (offs_h + 2 * H) * stride_hgh
    x_n = tl.load(x_n_ptrs, mask=mask, other=0.0)
    h_n = tl.load(h_n_ptrs, mask=mask, other=0.0)

    # n = tanh(x_n + r * h_n)
    n_pre = x_n + r * h_n
    t = tl.exp(-2.0 * n_pre)
    n = (1.0 - t) / (1.0 + t)  # tanh

    # previous hidden state
    h_prev_ptrs = h_prev_ptr + offs_b * stride_hpb + offs_h * stride_hph
    h_prev = tl.load(h_prev_ptrs, mask=mask, other=0.0)

    # h_t = (1 - z) * n + z * h_{t-1}
    h_new = (1.0 - z) * n + z * h_prev

    h_new_ptrs = h_new_ptr + offs_b * stride_hnb + offs_h * stride_hnh
    tl.store(h_new_ptrs, h_new, mask=mask)


def gru_step_triton(x_gates: torch.Tensor,
                    h_gates: torch.Tensor,
                    h_prev: torch.Tensor) -> torch.Tensor:
    """
    Single GRU step for one layer, over all batch elements.

    x_gates: [B, 3H] = x_t @ W_ih^T + b_ih, gate order [r, z, n]
    h_gates: [B, 3H] = h_{t-1} @ W_hh^T + b_hh, gate order [r, z, n]
    h_prev:  [B, H]
    returns: [B, H] (new hidden state)
    """
    assert x_gates.is_cuda and h_gates.is_cuda and h_prev.is_cuda
    B, threeH = x_gates.shape
    H = threeH // 3

    h_new = torch.empty((B, H), device=x_gates.device, dtype=x_gates.dtype)

    BLOCK_H = 128
    grid = (B, triton.cdiv(H, BLOCK_H))

    gru_step_kernel[grid](
        x_gates, h_gates, h_prev, h_new,
        B, H,
        x_gates.stride(0), x_gates.stride(1),
        h_gates.stride(0), h_gates.stride(1),
        h_prev.stride(0), h_prev.stride(1),
        h_new.stride(0), h_new.stride(1),
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )
    return h_new


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        Custom GRU implementation using Triton for the nonlinear gated update.

        IMPORTANT: This module reuses an internal nn.GRU (`self.gru`) so that
        its parameters (names, shapes, and layout) exactly match a standard
        nn.GRU. This allows state_dict loading/sharing with a reference model.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first

        # Reuse nn.GRU so parameter names & layout match reference exactly
        # (e.g., gru.weight_ih_l0, gru.weight_hh_l0, etc.).
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=False,
        )

    def forward(self, x, h0):
        """
        x:  [seq_len, batch_size, input_size] if batch_first=False
            [batch_size, seq_len, input_size] if batch_first=True
        h0: [num_layers, batch_size, hidden_size]
        Returns: output sequence (same layout as input, last dim = hidden_size)
        """
        if self.batch_first:
            # (batch, seq, feat) -> (seq, batch, feat)
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.shape
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h_prev_layers = [
                torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
        else:
            # h0: [num_layers, batch, hidden]
            assert h0.shape[0] == self.num_layers
            assert h0.shape[1] == batch_size
            assert h0.shape[2] == self.hidden_size
            h_prev_layers = [h0[layer] for layer in range(self.num_layers)]

        outputs = []

        # Iterate over time steps
        for t in range(seq_len):
            layer_input = x[t]  # [B, input_size or hidden_size]

            # Iterate over layers
            for layer in range(self.num_layers):
                # Fetch weights/biases from the internal nn.GRU so we
                # exactly mirror its parameters and state_dict layout.
                W_ih = getattr(self.gru, f"weight_ih_l{layer}")
                W_hh = getattr(self.gru, f"weight_hh_l{layer}")
                b_ih = getattr(self.gru, f"bias_ih_l{layer}") if self.bias else None
                b_hh = getattr(self.gru, f"bias_hh_l{layer}") if self.bias else None

                # Linear projections: [B, 3H] with gate order [r, z, n]
                x_gates = nn.functional.linear(layer_input, W_ih, b_ih)
                h_gates = nn.functional.linear(h_prev_layers[layer], W_hh, b_hh)

                # Triton kernel for gated update
                h_new = gru_step_triton(x_gates, h_gates, h_prev_layers[layer])

                h_prev_layers[layer] = h_new
                layer_input = h_new  # input to next layer

            outputs.append(layer_input)  # top layer output at this time step

        # Stack over time: [seq_len, batch, hidden]
        output = torch.stack(outputs, dim=0)

        if self.batch_first:
            # (seq, batch, feat) -> (batch, seq, feat)
            output = output.transpose(0, 1)

        # Match reference Model: return only output, not final hidden state
        return output
