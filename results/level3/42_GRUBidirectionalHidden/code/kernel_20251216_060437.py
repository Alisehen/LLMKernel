import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Conservative baseline
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_KX": 32,
                "BLOCK_KH": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        # More parallelism along batch
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 32,
                "BLOCK_KX": 32,
                "BLOCK_KH": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        # More parallelism along hidden
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_KX": 32,
                "BLOCK_KH": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["B", "H", "Kx"],
)
@triton.jit
def _gru_fused_gemm_pointwise_kernel(
    x_ptr,         # [B, Kx]
    h_prev_ptr,    # [B, H]
    w_ih_ptr,      # [3H, Kx]
    w_hh_ptr,      # [3H, H]
    b_ih_ptr,      # [3H]
    b_hh_ptr,      # [3H]
    h_new_ptr,     # [B, H]
    B, Kx, H,
    stride_xm, stride_xk,
    stride_hpm, stride_hpk,
    stride_wihm, stride_wihk,
    stride_whhm, stride_whhk,
    stride_bih,
    stride_bhh,
    stride_hnm, stride_hnk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KX: tl.constexpr,
    BLOCK_KH: tl.constexpr,
):
    """
    Fused GRU cell for a single time step:

        gates_x = x @ W_ih^T + b_ih        # [B, 3H]
        gates_h = h_prev @ W_hh^T + b_hh   # [B, 3H]

        reset  = sigmoid(i_r + h_r)
        update = sigmoid(i_z + h_z)
        new    = tanh(i_n + reset * h_n)
        h_new  = new + update * (h_prev - new)

    Tiling is over (batch=B, hidden=H). All GEMM + bias + pointwise
    are fused; the only global store is h_new.

    Optimized to reduce register pressure:
      - r, z gates keep merged input+hidden accumulators
      - n gate keeps separate input and hidden accumulators
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < B
    mask_n = offs_n < H
    mask_mn = mask_m[:, None] & mask_n[None, :]

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # ------------------------------------------------------------
    # Biases
    # Bias layout: [0:H] reset, [H:2H] update, [2H:3H] new
    # r, z: store combined (input + hidden) bias
    # n: store input and hidden separately
    # ------------------------------------------------------------
    bih_r = tl.load(
        b_ih_ptr + offs_n * stride_bih,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    bih_z = tl.load(
        b_ih_ptr + (offs_n + H) * stride_bih,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    bih_n = tl.load(
        b_ih_ptr + (offs_n + 2 * H) * stride_bih,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)

    bhh_r = tl.load(
        b_hh_ptr + offs_n * stride_bhh,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    bhh_z = tl.load(
        b_hh_ptr + (offs_n + H) * stride_bhh,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)
    bhh_n = tl.load(
        b_hh_ptr + (offs_n + 2 * H) * stride_bhh,
        mask=mask_n,
        other=0.0,
    ).to(tl.float32)

    # Combined / separated biases
    b_r = bih_r + bhh_r        # for reset gate: i_r + h_r
    b_z = bih_z + bhh_z        # for update gate: i_z + h_z
    b_in = bih_n               # new gate input contribution
    b_hn = bhh_n               # new gate hidden contribution

    # Broadcast across batch (M dimension)
    acc_r = tl.broadcast_to(b_r[None, :], (BLOCK_M, BLOCK_N)).to(tl.float32)
    acc_z = tl.broadcast_to(b_z[None, :], (BLOCK_M, BLOCK_N)).to(tl.float32)
    acc_in = tl.broadcast_to(b_in[None, :], (BLOCK_M, BLOCK_N)).to(tl.float32)
    acc_hn = tl.broadcast_to(b_hn[None, :], (BLOCK_M, BLOCK_N)).to(tl.float32)

    # ------------------------------------------------------------
    # GEMM: gates_x = x @ W_ih^T   (input contribution)
    # W_ih: [3H, Kx] (row-major)
    #   [0:H]   -> W_ir
    #   [H:2H]  -> W_iz
    #   [2H:3H] -> W_in
    # ------------------------------------------------------------
    kx = 0
    while kx < Kx:
        offs_kx = kx + tl.arange(0, BLOCK_KX)
        mask_kx = offs_kx < Kx

        # A tile: x[offs_m, offs_kx]  -> (BLOCK_M, BLOCK_KX)
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_kx[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_kx[None, :],
            other=0.0,
        ).to(tl.float32)

        # B tiles: W_ir, W_iz, W_in with shape (BLOCK_KX, BLOCK_N)
        w_ir = tl.load(
            w_ih_ptr
            + offs_n[None, :] * stride_wihm
            + offs_kx[:, None] * stride_wihk,
            mask=mask_kx[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        w_iz = tl.load(
            w_ih_ptr
            + (offs_n + H)[None, :] * stride_wihm
            + offs_kx[:, None] * stride_wihk,
            mask=mask_kx[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        w_in = tl.load(
            w_ih_ptr
            + (offs_n + 2 * H)[None, :] * stride_wihm
            + offs_kx[:, None] * stride_wihk,
            mask=mask_kx[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate input contributions
        acc_r += tl.dot(x_tile, w_ir, allow_tf32=True)
        acc_z += tl.dot(x_tile, w_iz, allow_tf32=True)
        acc_in += tl.dot(x_tile, w_in, allow_tf32=True)

        kx += BLOCK_KX

    # ------------------------------------------------------------
    # GEMM: gates_h = h_prev @ W_hh^T   (hidden contribution)
    # W_hh: [3H, H] (row-major)
    #   [0:H]   -> W_hr
    #   [H:2H]  -> W_hz
    #   [2H:3H] -> W_hn
    # ------------------------------------------------------------
    kh = 0
    while kh < H:
        offs_kh = kh + tl.arange(0, BLOCK_KH)
        mask_kh = offs_kh < H

        # A tile: h_prev[offs_m, offs_kh] -> (BLOCK_M, BLOCK_KH)
        h_tile = tl.load(
            h_prev_ptr
            + offs_m[:, None] * stride_hpm
            + offs_kh[None, :] * stride_hpk,
            mask=mask_m[:, None] & mask_kh[None, :],
            other=0.0,
        ).to(tl.float32)

        # B tiles: W_hr, W_hz, W_hn  -> (BLOCK_KH, BLOCK_N)
        w_hr = tl.load(
            w_hh_ptr
            + offs_n[None, :] * stride_whhm
            + offs_kh[:, None] * stride_whhk,
            mask=mask_kh[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        w_hz = tl.load(
            w_hh_ptr
            + (offs_n + H)[None, :] * stride_whhm
            + offs_kh[:, None] * stride_whhk,
            mask=mask_kh[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        w_hn = tl.load(
            w_hh_ptr
            + (offs_n + 2 * H)[None, :] * stride_whhm
            + offs_kh[:, None] * stride_whhk,
            mask=mask_kh[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate hidden contributions
        acc_r += tl.dot(h_tile, w_hr, allow_tf32=True)
        acc_z += tl.dot(h_tile, w_hz, allow_tf32=True)
        acc_hn += tl.dot(h_tile, w_hn, allow_tf32=True)

        kh += BLOCK_KH

    # ------------------------------------------------------------
    # Pointwise GRU equations (in registers)
    # ------------------------------------------------------------
    # h_prev tile for final mix
    h_prev_tile = tl.load(
        h_prev_ptr
        + offs_m[:, None] * stride_hpm
        + offs_n[None, :] * stride_hpk,
        mask=mask_mn,
        other=0.0,
    ).to(tl.float32)

    one = 1.0
    two = 2.0

    # resetgate = sigmoid(acc_r)
    r_lin = acc_r
    reset = one / (one + tl.exp(-r_lin))

    # updategate = sigmoid(acc_z)
    z_lin = acc_z
    update = one / (one + tl.exp(-z_lin))

    # newgate = tanh(acc_in + reset * acc_hn)
    pre_n = acc_in + reset * acc_hn
    newgate = two / (one + tl.exp(-two * pre_n)) - one

    # h_new = newgate + update * (h_prev - newgate)
    h_new = newgate + update * (h_prev_tile - newgate)

    # Single final store: h_new
    tl.store(
        h_new_ptr
        + offs_m[:, None] * stride_hnm
        + offs_n[None, :] * stride_hnk,
        h_new,
        mask=mask_mn,
    )


def gru_cell_fused_triton(
    x_t: torch.Tensor,   # [B, Kx]
    h_prev: torch.Tensor,  # [B, H]
    w_ih: torch.Tensor,  # [3H, Kx]
    w_hh: torch.Tensor,  # [3H, H]
    b_ih: torch.Tensor,  # [3H]
    b_hh: torch.Tensor,  # [3H]
) -> torch.Tensor:
    """
    Fused Triton GRU cell for a single time step:

        x_t:   [B, input_size or hidden_size * num_directions]
        h_prev:[B, H]
        w_ih:  [3H, Kx]
        w_hh:  [3H, H]
        b_ih:  [3H]
        b_hh:  [3H]

    Returns:
        h_new: [B, H]
    """
    assert x_t.is_cuda and h_prev.is_cuda
    assert w_ih.is_cuda and w_hh.is_cuda
    assert b_ih.is_cuda and b_hh.is_cuda

    B, Kx = x_t.shape
    B2, H = h_prev.shape
    assert B == B2, "Batch size mismatch between x_t and h_prev"

    threeH, Kx_w = w_ih.shape
    threeH2, H_w = w_hh.shape
    assert threeH == threeH2 == 3 * H
    assert Kx_w == Kx
    assert H_w == H
    assert b_ih.numel() == 3 * H
    assert b_hh.numel() == 3 * H

    # Ensure contiguous for optimal memory access
    x_t_c = x_t.contiguous()
    h_prev_c = h_prev.contiguous()
    w_ih_c = w_ih.contiguous()
    w_hh_c = w_hh.contiguous()
    b_ih_c = b_ih.contiguous()
    b_hh_c = b_hh.contiguous()

    h_new = torch.empty_like(h_prev_c)

    stride_xm, stride_xk = x_t_c.stride()
    stride_hpm, stride_hpk = h_prev_c.stride()
    stride_wihm, stride_wihk = w_ih_c.stride()
    stride_whhm, stride_whhk = w_hh_c.stride()
    stride_bih = b_ih_c.stride(0)
    stride_bhh = b_hh_c.stride(0)
    stride_hnm, stride_hnk = h_new.stride()

    def grid(meta):
        return (
            triton.cdiv(B, meta["BLOCK_M"]),
            triton.cdiv(H, meta["BLOCK_N"]),
        )

    _gru_fused_gemm_pointwise_kernel[grid](
        x_t_c,
        h_prev_c,
        w_ih_c,
        w_hh_c,
        b_ih_c,
        b_hh_c,
        h_new,
        B,
        Kx,
        H,
        stride_xm,
        stride_xk,
        stride_hpm,
        stride_hpk,
        stride_wihm,
        stride_wihk,
        stride_whhm,
        stride_whhk,
        stride_bih,
        stride_bhh,
        stride_hnm,
        stride_hnk,
    )

    return h_new


class ModelNew(nn.Module):
    """
    Triton-accelerated GRU implementation (bidirectional, multi-layer) that
    reuses nn.GRU parameters but executes each GRU cell with a single, fused
    high-performance Triton kernel (GEMM + bias + pointwise).

    This eliminates intermediate global stores of gates_x / gates_h:
    all intermediate values stay in registers until the final h_new store.
    """

    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=True,
        )
        self.num_directions = 2  # bidirectional=True

    # ---- helpers to access GRU parameters in the same layout as nn.GRU ----
    def _layer_direction_params(self, layer: int, direction: int):
        """
        Fetch (weight_ih, weight_hh, bias_ih, bias_hh) for given layer and direction
        from the internal nn.GRU so that parameters are exactly shared with the
        reference GRU.
        """
        suffix = "_reverse" if direction == 1 else ""
        w_ih = getattr(self.gru, f"weight_ih_l{layer}{suffix}")
        w_hh = getattr(self.gru, f"weight_hh_l{layer}{suffix}")

        if self.gru.bias:
            b_ih = getattr(self.gru, f"bias_ih_l{layer}{suffix}")
            b_hh = getattr(self.gru, f"bias_hh_l{layer}{suffix}")
        else:
            b_ih = None
            b_hh = None
        return w_ih, w_hh, b_ih, b_hh

    def _gru_cell(self, x_t, h_prev, w_ih, w_hh, b_ih, b_hh):
        """
        Single GRU cell step matching nn.GRU equations, with both gate
        linear parts and pointwise update fused into a single Triton kernel.

        x_t:   [B, input_size or hidden_size * num_directions]
        h_prev:[B, hidden_size]
        Returns: h_t: [B, hidden_size]
        """
        H = self.hidden_size

        # Ensure proper bias tensors (or zero if no bias)
        if b_ih is None:
            b_ih_eff = torch.zeros(3 * H, device=x_t.device, dtype=x_t.dtype)
        else:
            b_ih_eff = b_ih

        if b_hh is None:
            b_hh_eff = torch.zeros(3 * H, device=h_prev.device, dtype=h_prev.dtype)
        else:
            b_hh_eff = b_hh

        # Fused GEMM + bias + pointwise GRU update
        h_t = gru_cell_fused_triton(x_t, h_prev, w_ih, w_hh, b_ih_eff, b_hh_eff)
        return h_t

    def forward(self, x, h0):
        """
        x:  (seq_len, batch, input_size) if batch_first=False,
            (batch, seq_len, input_size) if batch_first=True
        h0: (num_layers * num_directions, batch, hidden_size)

        Returns:
            h_n: (num_layers * num_directions, batch, hidden_size)
        """
        # Arrange input as (seq_len, batch, input_size)
        if self.batch_first:
            x = x.transpose(0, 1)
        x = x.contiguous()

        seq_len, batch_size, _ = x.shape
        device = x.device
        dtype = x.dtype

        num_layers = self.num_layers
        num_directions = self.num_directions
        hidden_size = self.hidden_size

        if h0 is None:
            h_prev_all = torch.zeros(
                num_layers * num_directions,
                batch_size,
                hidden_size,
                device=device,
                dtype=dtype,
            )
        else:
            h_prev_all = h0

        # Output hidden states for all layers and directions
        h_n = torch.empty(
            num_layers * num_directions,
            batch_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )

        prev_layer_output = x  # [seq_len, batch, current_feature_size]

        for layer in range(num_layers):
            if layer == 0:
                layer_input_size = self.input_size
            else:
                layer_input_size = hidden_size * num_directions

            layer_outputs = []

            for direction in range(num_directions):
                if direction == 0:
                    time_indices = range(seq_len)
                else:
                    time_indices = range(seq_len - 1, -1, -1)

                w_ih, w_hh, b_ih, b_hh = self._layer_direction_params(layer, direction)
                hx = h_prev_all[layer * num_directions + direction]

                outputs_dir = []

                for t in time_indices:
                    x_t = prev_layer_output[t]  # [batch, layer_input_size]
                    hx = self._gru_cell(x_t, hx, w_ih, w_hh, b_ih, b_hh)
                    outputs_dir.append(hx)

                # Final hidden for this (layer, direction)
                h_n[layer * num_directions + direction] = hx

                # Stack outputs and align time order to [0..seq_len-1]
                outputs_dir = torch.stack(outputs_dir, dim=0)  # [seq_len, batch, hidden]
                if direction == 1:
                    # Reverse to match original time order
                    outputs_dir = torch.flip(outputs_dir, dims=[0])

                layer_outputs.append(outputs_dir)

            # Concatenate directions along feature dim: [seq_len, batch, hidden * num_directions]
            if num_directions == 1:
                prev_layer_output = layer_outputs[0]
            else:
                prev_layer_output = torch.cat(layer_outputs, dim=2)

        return h_n
