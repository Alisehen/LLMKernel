import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Conservative baseline: lower register pressure, good general performance
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=2,
        ),
        # More aggressive: higher parallelism & deeper pipelining for compute-bound cases
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=8,
            num_stages=3,
        ),
        # Rectangular tile in N: helps when N is small or very skinny
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8},
            num_warps=4,
            num_stages=2,
        ),
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
    APPLY_TANH: tl.constexpr,  # compile-time specialization
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    # ------------------------------
    # Program ID mapping with M-grouping for better L2 reuse
    # ------------------------------
    pid = tl.program_id(axis=0)

    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    group_m = GROUP_M
    num_pid_in_group = group_m * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_m)
    pid_n = pid_in_group // group_m

    # Early exit for out-of-bounds program IDs
    if pid_m >= num_pid_m:
        return

    # ------------------------------
    # Compute tile indices
    # ------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m[:, None] < M        # (BLOCK_M, 1)
    mask_n = offs_n[None, :] < N        # (1, BLOCK_N)

    # Base pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in fp32 to keep numerical stability and match PyTorch Linear
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------------------------
    # K loop
    # ------------------------------
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

        # Use tensor cores where possible, accumulate in fp32
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ------------------------------
    # Bias add (broadcast over M)
    # ------------------------------
    bias_mask = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)  # (BLOCK_N,)
    acc += bias[None, :]

    # ------------------------------
    # Optional tanh activation
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # ------------------------------
    if APPLY_TANH:
        acc = tl.exp(acc * 2.0)
        acc = (acc - 1.0) / (acc + 1.0)

    # ------------------------------
    # Write back
    # ------------------------------
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = mask_m & mask_n
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

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

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
