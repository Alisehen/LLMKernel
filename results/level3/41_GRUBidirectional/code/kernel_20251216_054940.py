import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------
# Low-level math helpers
# ----------------------------

@triton.jit
def _sigmoid(x):
    # sigmoid(x) = 1 / (1 + exp(-x))
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _tanh_from_sigmoid(x):
    # tanh(x) = 2 * sigmoid(2x) - 1
    return 2.0 * _sigmoid(2.0 * x) - 1.0


# ----------------------------
# Matmul + Bias kernel (A @ B + bias)
# A: (M, K), B: (K, N), bias: (N,), C: (M, N)
# ----------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_stages=4,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
            },
            num_stages=3,
            num_warps=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_bias_kernel(
    a_ptr,  # *const A, shape (M, K)
    b_ptr,  # *const B, shape (K, N)
    bias_ptr,  # *const bias, shape (N,)
    c_ptr,  # *C, shape (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B + bias
    All fused ops (matmul + bias) share the SAME (offs_m, offs_n) and masks.
    Grid: 2D over tiles of C.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program in the M and N dimensions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Boundary masks (shared by all fused ops)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Create pointers for A and B tiles
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulate in fp32 for numerical stability / tensor cores
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        k_mask_broadcast_m = k_mask[None, :]
        k_mask_broadcast_n = k_mask[:, None]

        # Load A and B tiles with proper masking
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask_broadcast_m,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask_broadcast_n & mask_n[None, :],
            other=0.0,
        )

        # FMA on tensor cores when possible
        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fuse bias add using the SAME offs_n and mask_n as matmul
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Write C
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc.to(tl.type_of(tl.load(c_ptr, mask=False, other=0.0))),
        mask=mask_m[:, None] & mask_n[None, :],
    )


# ----------------------------
# GRU elementwise gate kernel
# r = sigmoid(i_r + h_r)
# z = sigmoid(i_z + h_z)
# n = tanh(i_n + r * h_n)
# h_new = (1 - z) * n + z * h_prev
#
# gates_x, gates_h: (B, 3H)
# h_prev, h_new, out_t: (B, H)
# ----------------------------

@triton.jit
def gru_gates_kernel(
    gates_x_ptr,  # (B, 3H)
    gates_h_ptr,  # (B, 3H)
    h_prev_ptr,   # (B, H)
    h_new_ptr,    # (B, H)
    out_ptr,      # (B, H)  - slice output[t]
    B, H,
    stride_gx_b, stride_gx_c,
    stride_gh_b, stride_gh_c,
    stride_hp_b, stride_hp_c,
    stride_hn_b, stride_hn_c,
    stride_out_b, stride_out_c,
    BLOCK: tl.constexpr,
):
    """
    Elementwise GRU cell update kernel.
    All fused ops share the SAME 1D offsets / mask.
    Grid: 1D over elements of (B, H) flattened.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = B * H
    mask = offs < total

    # Map 1D offsets -> (b, h) using the SAME offsets for all fused ops
    b_idx = offs // H
    h_idx = offs % H

    # Compute column indices for each gate slice
    col_r = h_idx
    col_z = h_idx + H
    col_n = h_idx + 2 * H

    # Base pointers
    gx_ir_ptrs = gates_x_ptr + b_idx * stride_gx_b + col_r * stride_gx_c
    gx_iz_ptrs = gates_x_ptr + b_idx * stride_gx_b + col_z * stride_gx_c
    gx_in_ptrs = gates_x_ptr + b_idx * stride_gx_b + col_n * stride_gx_c

    gh_hr_ptrs = gates_h_ptr + b_idx * stride_gh_b + col_r * stride_gh_c
    gh_hz_ptrs = gates_h_ptr + b_idx * stride_gh_b + col_z * stride_gh_c
    gh_hn_ptrs = gates_h_ptr + b_idx * stride_gh_b + col_n * stride_gh_c

    hp_ptrs = h_prev_ptr + b_idx * stride_hp_b + h_idx * stride_hp_c
    hn_ptrs = h_new_ptr + b_idx * stride_hn_b + h_idx * stride_hn_c
    out_h_ptrs = out_ptr + b_idx * stride_out_b + h_idx * stride_out_c

    # Load inputs
    i_r = tl.load(gx_ir_ptrs, mask=mask, other=0.0)
    i_z = tl.load(gx_iz_ptrs, mask=mask, other=0.0)
    i_n = tl.load(gx_in_ptrs, mask=mask, other=0.0)

    h_r = tl.load(gh_hr_ptrs, mask=mask, other=0.0)
    h_z = tl.load(gh_hz_ptrs, mask=mask, other=0.0)
    h_n = tl.load(gh_hn_ptrs, mask=mask, other=0.0)

    h_prev = tl.load(hp_ptrs, mask=mask, other=0.0)

    # Fused gate computations, all using the SAME offsets / mask
    r = _sigmoid(i_r + h_r)
    z = _sigmoid(i_z + h_z)
    n = _tanh_from_sigmoid(i_n + r * h_n)
    h_new = (1.0 - z) * n + z * h_prev

    # Store results: next hidden and output[t]
    tl.store(hn_ptrs, h_new, mask=mask)
    tl.store(out_h_ptrs, h_new, mask=mask)


# ----------------------------
# Wrapper: matmul + bias
# ----------------------------

def linear_triton(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Compute y = x @ weight + bias using an optimized Triton kernel.
    x:        (M, K)
    weight_t: (K, N)  -- transposed weight
    bias:     (N,)
    Returns y: (M, N)
    """
    assert x.is_cuda and weight_t.is_cuda and bias.is_cuda
    assert x.dtype == weight_t.dtype == bias.dtype

    M, K = x.shape
    K_w, N = weight_t.shape
    assert K_w == K

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    stride_am, stride_ak = x.stride()
    stride_bk, stride_bn = weight_t.stride()
    stride_cm, stride_cn = y.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_bias_kernel[grid](
        x,
        weight_t,
        bias,
        y,
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
    return y


# ----------------------------
# Wrapper: GRU gates elementwise
# ----------------------------

def gru_cell_triton(
    gates_x: torch.Tensor,  # (B, 3H)
    gates_h: torch.Tensor,  # (B, 3H)
    h_prev: torch.Tensor,   # (B, H)
    out_t: torch.Tensor,    # (B, H) slice of output[t]
    work_buf: torch.Tensor, # (B, H) pre-allocated buffer for h_new
) -> torch.Tensor:
    """
    Fused elementwise GRU cell using Triton.
    All tensors must be CUDA and same dtype.
    """
    assert gates_x.is_cuda and gates_h.is_cuda and h_prev.is_cuda
    assert out_t.is_cuda and work_buf.is_cuda
    assert gates_x.dtype == gates_h.dtype == h_prev.dtype == out_t.dtype == work_buf.dtype

    B, threeH = gates_x.shape
    B2, H = h_prev.shape
    assert B == B2 and threeH == 3 * H
    assert gates_h.shape == gates_x.shape
    assert out_t.shape == h_prev.shape
    assert work_buf.shape == h_prev.shape

    stride_gx_b, stride_gx_c = gates_x.stride()
    stride_gh_b, stride_gh_c = gates_h.stride()
    stride_hp_b, stride_hp_c = h_prev.stride()
    stride_hn_b, stride_hn_c = work_buf.stride()
    stride_out_b, stride_out_c = out_t.stride()

    total = B * H
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)

    gru_gates_kernel[grid](
        gates_x,
        gates_h,
        h_prev,
        work_buf,
        out_t,
        B,
        H,
        stride_gx_b,
        stride_gx_c,
        stride_gh_b,
        stride_gh_c,
        stride_hp_b,
        stride_hp_c,
        stride_hn_b,
        stride_hn_c,
        stride_out_b,
        stride_out_c,
        BLOCK=BLOCK,
    )

    return work_buf


# ----------------------------
# High-level GRU layer & network forward
# ----------------------------

def gru_layer_triton(
    x: torch.Tensor,          # (seq_len, batch, input_size_or_2H)
    h0: torch.Tensor,         # (batch, hidden_size)
    w_ih_t: torch.Tensor,     # (input_size_or_2H, 3*hidden_size)  transposed
    b_ih: torch.Tensor,       # (3*hidden_size,)
    w_hh_t: torch.Tensor,     # (hidden_size, 3*hidden_size)       transposed
    b_hh: torch.Tensor,       # (3*hidden_size,)
    reverse: bool,
) -> (torch.Tensor, torch.Tensor):
    """
    Single GRU layer for one direction, using Triton for both
    linear parts and gate elementwise computations.
    """
    seq_len, batch_size, _ = x.shape
    hidden_size = h0.shape[1]
    device = x.device
    dtype = x.dtype

    output = torch.empty((seq_len, batch_size, hidden_size), device=device, dtype=dtype)

    # Work buffers for hidden state to avoid per-step allocations
    h_t = h0.contiguous()
    h_next = torch.empty_like(h_t)

    if reverse:
        time_range = range(seq_len - 1, -1, -1)
    else:
        time_range = range(seq_len)

    for t in time_range:
        x_t = x[t]  # (batch, input_size_or_2H)

        # GEMMs on x_t and h_t (fused with bias inside kernel)
        gates_x = linear_triton(x_t, w_ih_t, b_ih)  # (batch, 3H)
        gates_h = linear_triton(h_t, w_hh_t, b_hh)  # (batch, 3H)

        # Fused GRU gate + hidden update in one Triton kernel
        out_t_slice = output[t]  # (batch, H) view
        h_next = gru_cell_triton(gates_x, gates_h, h_t, out_t_slice, h_next)

        # Swap buffers
        h_t, h_next = h_next, h_t

    return output, h_t


def gru_triton_forward(
    x: torch.Tensor,
    h0: torch.Tensor,
    gru_module: nn.GRU,
) -> (torch.Tensor, torch.Tensor):
    """
    Multi-layer (possibly bidirectional) GRU forward using Triton
    for all linear and GRU-cell elementwise operations.
    x: (seq_len, batch, input_size) if not batch_first
       or (batch, seq_len, input_size) if batch_first
    h0: (num_layers * num_directions, batch, hidden_size)
    """
    batch_first = getattr(gru_module, "batch_first", False)
    if batch_first:
        x = x.transpose(0, 1)  # (seq, batch, feat)

    seq_len, batch_size, _ = x.shape
    num_layers = gru_module.num_layers
    hidden_size = gru_module.hidden_size
    num_directions = 2 if gru_module.bidirectional else 1
    use_bias = gru_module.bias

    assert h0.shape[0] == num_layers * num_directions
    assert h0.shape[1] == batch_size
    assert h0.shape[2] == hidden_size

    current_input = x
    h_n_list = []

    for layer in range(num_layers):
        layer_outputs = []

        for direction in range(num_directions):
            suffix = "" if direction == 0 else "_reverse"
            reverse = direction == 1

            w_ih = getattr(gru_module, f"weight_ih_l{layer}{suffix}")
            w_hh = getattr(gru_module, f"weight_hh_l{layer}{suffix}")

            if use_bias:
                b_ih = getattr(gru_module, f"bias_ih_l{layer}{suffix}")
                b_hh = getattr(gru_module, f"bias_hh_l{layer}{suffix}")
            else:
                b_ih = torch.zeros(
                    3 * hidden_size, device=current_input.device, dtype=current_input.dtype
                )
                b_hh = torch.zeros(
                    3 * hidden_size, device=current_input.device, dtype=current_input.dtype
                )

            # Transpose weights once per forward pass for Triton kernel
            w_ih_t = w_ih.transpose(0, 1).contiguous()  # (input_size, 3H)
            w_hh_t = w_hh.transpose(0, 1).contiguous()  # (H, 3H)

            h0_ld = h0[layer * num_directions + direction]  # (batch, H)

            out_dir, h_n_dir = gru_layer_triton(
                current_input, h0_ld, w_ih_t, b_ih, w_hh_t, b_hh, reverse
            )
            layer_outputs.append(out_dir)
            h_n_list.append(h_n_dir.unsqueeze(0))

        if num_directions == 1:
            current_input = layer_outputs[0]
        else:
            current_input = torch.cat(layer_outputs, dim=2)  # concat directions on feature dim

    output = current_input
    h_n = torch.cat(h_n_list, dim=0)  # (num_layers * num_directions, batch, H)

    if batch_first:
        output = output.transpose(0, 1)  # back to (batch, seq, feat)

    return output, h_n


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        GRU model using Triton-accelerated kernels for GEMM and GRU cell math.
        """
        super(ModelNew, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=0.0,
            bidirectional=True,
        )

    def forward(self, x, h0):
        output, _ = gru_triton_forward(x, h0, self.gru)
        return output
