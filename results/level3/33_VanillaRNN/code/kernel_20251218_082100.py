import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Smaller tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        # Medium tiles
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        # Larger tiles
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_activation_kernel(
    a_ptr,      # (M, K)
    b_ptr,      # (K, N)
    bias_ptr,   # (N,)
    c_ptr,      # (M, N)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    APPLY_TANH: tl.constexpr,  # whether to apply tanh or not
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id: tiles over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for M and N dimensions (broadcast in later ops)
    mask_m = offs_m[:, None] < M          # (BLOCK_M, 1)
    mask_n = offs_n[None, :] < N          # (1, BLOCK_N)

    # Base pointers for this tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_mask = offs_k < k_remaining  # (BLOCK_K,)

        a = tl.load(
            a_ptrs,
            mask=mask_m & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & mask_n,
            other=0.0,
        )

        a_f32 = a.to(tl.float32)
        b_f32 = b.to(tl.float32)

        # Use tensor cores where possible (allow_tf32)
        acc += tl.dot(a_f32, b_f32, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Output pointers and mask
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = mask_m & mask_n  # (BLOCK_M, BLOCK_N)

    # Fused bias add: broadcast bias[None, :]
    bias_mask = offs_n < N  # 1D mask for bias load; avoids unsupported indexing
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)  # (BLOCK_N,)
    acc += bias[None, :]

    # Optional tanh activation
    if APPLY_TANH:
        z = acc * 2.0
        exp2z = tl.exp(z)
        acc = (exp2z - 1.0) / (exp2z + 1.0)

    # Store result
    tl.store(c_ptrs, acc, mask=out_mask)


def _fused_linear_kernel_launch(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor,
    apply_tanh: bool,
) -> torch.Tensor:
    """
    x:        (M, K), contiguous
    weight_t: (K, N), contiguous (transpose of original weight)
    bias:     (N,)
    returns:  (M, N) in same dtype as x
    """
    assert x.is_cuda and weight_t.is_cuda and bias.is_cuda
    assert x.dtype == weight_t.dtype == bias.dtype

    M, K = x.shape
    K_w, N = weight_t.shape
    assert K_w == K

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

    linear_bias_activation_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        out.stride(0), out.stride(1),
        APPLY_TANH=apply_tanh,
    )
    return out


def fused_linear_tanh(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      (M, K)
    weight: (N, K)  - PyTorch Linear weight (out_features, in_features)
    bias:   (N,)
    returns (M, N) with Tanh applied
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype

    x_contig = x.contiguous()
    w = weight.contiguous()          # (N, K)
    w_t = w.t().contiguous()         # (K, N)

    return _fused_linear_kernel_launch(x_contig, w_t, bias, apply_tanh=True)


def fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      (M, K)
    weight: (N, K)  - PyTorch Linear weight (out_features, in_features)
    bias:   (N,)
    returns (M, N) without activation
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype

    x_contig = x.contiguous()
    w = weight.contiguous()
    w_t = w.t().contiguous()

    return _fused_linear_kernel_launch(x_contig, w_t, bias, apply_tanh=False)


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Vanilla RNN with Triton-fused Linear + Tanh and Linear layers.
        Semantics match the original Model.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Keep nn.Linear modules so state_dict is compatible with the original Model
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()  # structural compatibility only

        # Lazy-initialized hidden state buffer
        self.register_buffer("hidden", None, persistent=False)

    def forward(self, x: torch.Tensor, initial_hidden: torch.Tensor = None) -> torch.Tensor:
        """
        x:              (batch_size, input_size)
        initial_hidden: (batch_size, hidden_size) or None
        returns:        (batch_size, output_size)
        """
        assert x.dim() == 2
        batch_size = x.shape[0]

        if initial_hidden is not None:
            hidden = initial_hidden
        else:
            if (self.hidden is None) or (self.hidden.shape[0] != batch_size):
                self.hidden = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            hidden = self.hidden

        hidden = hidden.to(device=x.device, dtype=x.dtype)

        # Concatenate input and previous hidden state
        combined = torch.cat((x, hidden), dim=1)

        # Fused input-to-hidden linear + bias + tanh
        new_hidden = fused_linear_tanh(combined, self.i2h.weight, self.i2h.bias)

        # Fused hidden-to-output linear + bias
        output = fused_linear(new_hidden, self.h2o.weight, self.h2o.bias)

        # Update internal hidden buffer (detach to avoid graph retention)
        self.hidden = new_hidden.detach()

        return output
