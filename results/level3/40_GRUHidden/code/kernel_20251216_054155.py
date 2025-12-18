import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton MatMul Kernel: C[M, N] = A[M, K] @ B[K, N]
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
            num_stages=2,
            num_warps=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for C tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write back
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    c = acc
    tl.store(c_ptrs, c, mask=c_mask)


# ---------------------------------------------------------------------------
# Python Wrapper for Triton MatMul
# ---------------------------------------------------------------------------

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ B using Triton.
    A: [M, K]
    B: [K, N]
    Returns C: [M, N]
    """
    assert a.ndim == 2 and b.ndim == 2, "triton_matmul only supports 2D tensors"
    assert a.shape[1] == b.shape[0], "Incompatible matmul shapes"

    if not a.is_cuda or not b.is_cuda:
        raise ValueError("triton_matmul expects CUDA tensors")

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


# ---------------------------------------------------------------------------
# Model with GRU implemented using Triton matmul
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        Custom GRU model using Triton matmul for gate computations.
        Parameters mirror nn.GRU so that state_dicts are compatible.
        """
        super(ModelNew, self).__init__()
        # Use an internal nn.GRU purely to hold parameters with the exact same
        # shapes and names as a standard GRU. We will NOT use its forward.
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
        )

    def forward(self, x: torch.Tensor, h0: torch.Tensor):
        """
        x: (seq_len, batch, input_size) if batch_first=False
           (batch, seq_len, input_size) if batch_first=True
        h0: (num_layers, batch, hidden_size)
        Returns:
            h_n: (num_layers, batch, hidden_size)
        """
        batch_first = self.gru.batch_first
        num_layers = self.gru.num_layers
        hidden_size = self.gru.hidden_size
        bias = self.gru.bias

        if batch_first:
            x = x.transpose(0, 1).contiguous()  # (seq, batch, input)

        seq_len, batch_size, _ = x.shape
        assert h0.shape[0] == num_layers
        assert h0.shape[1] == batch_size
        assert h0.shape[2] == hidden_size

        layer_input = x
        h_n_list = []

        for layer in range(num_layers):
            # Fetch weights & biases from the internal nn.GRU
            weight_ih = getattr(self.gru, f"weight_ih_l{layer}")  # (3H, in_dim)
            weight_hh = getattr(self.gru, f"weight_hh_l{layer}")  # (3H, H)

            b_ih = getattr(self.gru, f"bias_ih_l{layer}") if bias else None  # (3H,)
            b_hh = getattr(self.gru, f"bias_hh_l{layer}") if bias else None  # (3H,)

            # Transpose weights once per layer to match x @ W layout
            W_ih_T = weight_ih.transpose(0, 1).contiguous()  # (in_dim, 3H)
            W_hh_T = weight_hh.transpose(0, 1).contiguous()  # (H, 3H)

            # Hidden state for this layer
            h_t = h0[layer]  # (batch, H)

            # Output for this layer
            layer_output = torch.empty(
                (seq_len, batch_size, hidden_size),
                device=x.device,
                dtype=x.dtype,
            )

            for t in range(seq_len):
                x_t = layer_input[t]  # (batch, in_dim)

                # gate_x = x_t @ W_ih^T -> use W_ih_T
                gate_x = triton_matmul(x_t, W_ih_T)
                if b_ih is not None:
                    gate_x = gate_x + b_ih

                # gate_h = h_t @ W_hh^T -> use W_hh_T
                gate_h = triton_matmul(h_t, W_hh_T)
                if b_hh is not None:
                    gate_h = gate_h + b_hh

                i_r, i_z, i_n = gate_x.chunk(3, dim=1)
                h_r, h_z, h_n = gate_h.chunk(3, dim=1)

                r = torch.sigmoid(i_r + h_r)
                z = torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)

                h_t = (1.0 - z) * n + z * h_t
                layer_output[t] = h_t

            h_n_list.append(h_t)
            layer_input = layer_output

        h_n = torch.stack(h_n_list, dim=0)  # (num_layers, batch, hidden_size)
        return h_n
