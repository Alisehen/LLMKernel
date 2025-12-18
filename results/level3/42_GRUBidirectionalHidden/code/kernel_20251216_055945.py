import torch
import torch.nn as nn
import triton
import triton.language as tl


# -------------------------------
# GEMM + Bias kernel (optimized)
# -------------------------------

@triton.autotune(
    configs=[
        # Good defaults for Ada / 4090 across a wide range of shapes
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=2,
        ),
        # Larger K tile for very tall K
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_gemm_bias_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N] (weight is [N, K] but passed as-is; kernel sees [K, N])
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute C = A @ B + bias
    A: [M, K]
    B: [K, N]
    bias: [N]
    C: [M, N]

    Grid: 1D over tiles of (M, N) using grouped ordering for L2 reuse.
    All fused ops (matmul, bias, store) share the same (offs_m, offs_n, mask_mn).
    """
    pid = tl.program_id(axis=0)

    # 2D tile indices with grouped ordering along M for better L2 locality
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = grid_m
    num_pid_n = grid_n

    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % (group_size_m * num_pid_n)
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_mn = mask_m[:, None] & mask_n[None, :]

    # Hints for better codegen
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    offs_k = tl.arange(0, BLOCK_K)
    tl.multiple_of(offs_k, BLOCK_K)

    # Pointers for first K-tile
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    k = 0
    while k < K:
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Fused bias add along N (broadcast over M)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=mask_mn)


# -------------------------------
# GRU pointwise (gates + update)
# -------------------------------

@triton.jit
def _gru_pointwise_kernel(
    gates_x_ptr,  # [B, 3H]
    gates_h_ptr,  # [B, 3H]
    h_prev_ptr,   # [B, H]
    h_new_ptr,    # [B, H]
    B, H,
    stride_gxm, stride_gxn,
    stride_ghm, stride_ghn,
    stride_hpm, stride_hpn,
    stride_hnm, stride_hnn,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused pointwise GRU cell update for a single time step:

        i_r, i_z, i_n from gates_x  (input)
        h_r, h_z, h_n from gates_h  (hidden)
        h_prev

        resetgate = sigmoid(i_r + h_r)
        updategate = sigmoid(i_z + h_z)
        newgate = tanh(i_n + resetgate * h_n)
        h_new = newgate + updategate * (h_prev - newgate)

    All operations are elementwise with a 1D grid over [B * H].
    All loads/stores share the same flat offsets & mask.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = B * H
    mask = offs < total

    # Map flat index -> (batch, hidden)
    batch = offs // H
    hid = offs % H

    # Base indices into gates_x / gates_h (for reset gate)
    gx_base = gates_x_ptr + batch * stride_gxm + hid * stride_gxn
    gh_base = gates_h_ptr + batch * stride_ghm + hid * stride_ghn

    # Input gates (i_r, i_z, i_n)
    i_r = tl.load(gx_base, mask=mask, other=0.0).to(tl.float32)
    i_z = tl.load(gx_base + H * stride_gxn, mask=mask, other=0.0).to(tl.float32)
    i_n = tl.load(gx_base + 2 * H * stride_gxn, mask=mask, other=0.0).to(tl.float32)

    # Hidden gates (h_r, h_z, h_n)
    h_r = tl.load(gh_base, mask=mask, other=0.0).to(tl.float32)
    h_z = tl.load(gh_base + H * stride_ghn, mask=mask, other=0.0).to(tl.float32)
    h_n = tl.load(gh_base + 2 * H * stride_ghn, mask=mask, other=0.0).to(tl.float32)

    # Previous hidden
    h_prev = tl.load(
        h_prev_ptr + batch * stride_hpm + hid * stride_hpn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    # resetgate = sigmoid(i_r + h_r)
    pre_r = i_r + h_r
    reset = 1.0 / (1.0 + tl.exp(-pre_r))

    # updategate = sigmoid(i_z + h_z)
    pre_z = i_z + h_z
    update = 1.0 / (1.0 + tl.exp(-pre_z))

    # newgate = tanh(i_n + reset * h_n)
    pre_n = i_n + reset * h_n
    # tanh(x) = 2 * sigmoid(2x) - 1
    newgate = 2.0 / (1.0 + tl.exp(-2.0 * pre_n)) - 1.0

    # h_new = newgate + update * (h_prev - newgate)
    h_new = newgate + update * (h_prev - newgate)

    tl.store(
        h_new_ptr + batch * stride_hnm + hid * stride_hnn,
        h_new,
        mask=mask,
    )


# -------------------------------
# Python wrappers
# -------------------------------

def linear_triton(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Triton-accelerated replacement for torch.nn.functional.linear:

        out = input @ weight.T + bias

    input:  [M, K]
    weight: [N, K]
    bias:   [N]
    out:    [M, N]
    """
    assert input.ndim == 2
    assert weight.ndim == 2
    assert bias is not None and bias.ndim == 1

    M, K = input.shape
    N, K_w = weight.shape
    assert K == K_w

    assert input.device == weight.device == bias.device
    assert input.is_cuda

    out = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # Strides for input [M, K]
    stride_am, stride_ak = input.stride()
    # Strides for weight [N, K] but viewed as [K, N] inside the kernel
    stride_wn, stride_wk = weight.stride()
    stride_bk = stride_wk  # along K
    stride_bn = stride_wn  # along N
    # Strides for output [M, N]
    stride_cm, stride_cn = out.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )

    _linear_gemm_bias_kernel[grid](
        input,
        weight,
        bias,
        out,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
    )
    return out


def gru_pointwise_triton(gates_x: torch.Tensor,
                         gates_h: torch.Tensor,
                         h_prev: torch.Tensor) -> torch.Tensor:
    """
    Fused Triton kernel for GRU pointwise equations (single time step).

    gates_x: [B, 3H] from input linear
    gates_h: [B, 3H] from hidden linear
    h_prev:  [B, H]
    returns: [B, H] h_new
    """
    assert gates_x.ndim == 2 and gates_h.ndim == 2 and h_prev.ndim == 2
    B, threeH = gates_x.shape
    B2, threeH2 = gates_h.shape
    B3, H = h_prev.shape
    assert B == B2 == B3
    assert threeH == threeH2
    assert threeH % 3 == 0
    assert gates_x.device == gates_h.device == h_prev.device
    assert gates_x.is_cuda and gates_h.is_cuda and h_prev.is_cuda

    H = threeH // 3
    out = torch.empty_like(h_prev)

    stride_gxm, stride_gxn = gates_x.stride()
    stride_ghm, stride_ghn = gates_h.stride()
    stride_hpm, stride_hpn = h_prev.stride()
    stride_hnm, stride_hnn = out.stride()

    total = B * H

    def grid(meta):
        return (triton.cdiv(total, meta["BLOCK_SIZE"]),)

    _gru_pointwise_kernel[grid](
        gates_x,
        gates_h,
        h_prev,
        out,
        B,
        H,
        stride_gxm,
        stride_gxn,
        stride_ghm,
        stride_ghn,
        stride_hpm,
        stride_hpn,
        stride_hnm,
        stride_hnn,
        BLOCK_SIZE=256,
    )
    return out


# -------------------------------
# GRU module using Triton kernels
# -------------------------------

class ModelNew(nn.Module):
    """
    Triton-accelerated GRU implementation (bidirectional, multi-layer) that
    reuses the exact nn.GRU parameters so it can share/load the reference
    GRU weights and produce numerically matching outputs.
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
        self.num_directions = 2  # because bidirectional=True

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
        Single GRU cell step matching nn.GRU equations, with gate linear
        parts computed via Triton GEMM and pointwise via Triton elementwise kernel.

        x_t:   [B, input_size or hidden_size * num_directions]
        h_prev:[B, hidden_size]
        Returns h_t: [B, hidden_size]
        """
        H = self.hidden_size
        B = x_t.shape[0]

        # gates from input: [B, 3*H]
        if b_ih is None:
            b_ih_eff = torch.zeros(3 * H, device=x_t.device, dtype=x_t.dtype)
        else:
            b_ih_eff = b_ih

        gates_x = linear_triton(x_t, w_ih, b_ih_eff)

        # gates from hidden: [B, 3*H]
        if b_hh is None:
            b_hh_eff = torch.zeros(3 * H, device=h_prev.device, dtype=h_prev.dtype)
        else:
            b_hh_eff = b_hh

        gates_h = linear_triton(h_prev, w_hh, b_hh_eff)

        # Fused pointwise GRU update in Triton
        h_t = gru_pointwise_triton(gates_x, gates_h, h_prev)
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
