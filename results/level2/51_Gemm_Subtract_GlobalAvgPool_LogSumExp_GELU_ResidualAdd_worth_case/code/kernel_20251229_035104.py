import math
import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def row_dot_gelu_kernel(
    x_ptr, c_ptr, out_ptr,
    M, K,
    stride_xm, stride_xk,
    scale, bias_term,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # row id
    offs_k = tl.arange(0, BLOCK_K)

    acc = 0.0

    # loop over feature dimension K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        mask = (pid_m < M) & (k_offsets < K)

        x_vals = tl.load(
            x_ptr + pid_m * stride_xm + k_offsets * stride_xk,
            mask=mask,
            other=0.0
        )
        c_vals = tl.load(
            c_ptr + k_offsets,
            mask=k_offsets < K,
            other=0.0
        )

        acc += tl.sum(x_vals * c_vals, axis=0)

    # mean over out_features and add bias/subtract contributions
    val = acc * scale + bias_term

    # GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
    x3 = val * val * val
    inner = 0.7978845608028654 * (val + 0.044715 * x3)
    exp2i = tl.exp(2.0 * inner)
    tanh_inner = (exp2i - 1.0) / (exp2i + 1.0)
    gelu_val = 0.5 * val * (1.0 + tanh_inner)

    tl.store(out_ptr + pid_m, gelu_val, mask=pid_m < M)


def fused_row_gelu(x, weight, bias, subtract):
    """
    Computes per-row scalar:
        s_m = GELU( mean_n( (x @ weight + bias - subtract)[m, n] ) )
    using algebraic simplification:
        mean_n( xW + b - s ) = (x_m · c + sum(b) - sum(s)) / N,
        where c_k = sum_n W_{k,n}, N = out_features.
    """
    assert x.is_cuda, "Input must be on CUDA device"
    x = x.contiguous()
    weight = weight.contiguous()
    subtract = subtract.contiguous()

    M, K = x.shape
    K_w, N = weight.shape
    assert K_w == K, "Weight shape mismatch"

    # Precompute column sums and constants on the same device
    c = weight.sum(dim=1)  # (K,)
    if bias is not None:
        bias_sum = bias.sum()
    else:
        bias_sum = torch.zeros((), device=x.device, dtype=x.dtype)
    sub_sum = subtract.sum()

    scale = 1.0 / float(N)
    bias_term = (bias_sum - sub_sum) * scale

    out = torch.empty((M,), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, 1),)
    row_dot_gelu_kernel[grid](
        x, c, out,
        M, K,
        x.stride(0), x.stride(1),
        scale, bias_term,
        BLOCK_K=256,
    )
    return out


@triton.jit
def residual_add_kernel(
    orig_ptr, row_ptr, out_ptr,
    M, K,
    stride_om, stride_ok,
    stride_outm, stride_outk,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m < M) & (offs_n < K)

    orig_vals = tl.load(
        orig_ptr + offs_m * stride_om + offs_n * stride_ok,
        mask=mask,
        other=0.0
    )

    row_val = tl.load(
        row_ptr + offs_m,
        mask=offs_m < M,
        other=0.0
    )

    out_vals = orig_vals + row_val

    tl.store(
        out_ptr + offs_m * stride_outm + offs_n * stride_outk,
        out_vals,
        mask=mask
    )


def residual_add(original_x, row_scalars):
    """
    Broadcast-add row_scalars (M,) to original_x (M, K):
        out[m, k] = original_x[m, k] + row_scalars[m]
    """
    assert original_x.is_cuda and row_scalars.is_cuda
    original_x = original_x.contiguous()
    row_scalars = row_scalars.contiguous()

    M, K = original_x.shape
    out = torch.empty_like(original_x)

    grid = lambda META: (
        triton.cdiv(M, 1),
        triton.cdiv(K, META['BLOCK_N']),
    )
    residual_add_kernel[grid](
        original_x, row_scalars, out,
        M, K,
        original_x.stride(0), original_x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=256,
    )
    return out


class ModelNew(nn.Module):
    """
    High-performance Triton implementation of:

        x0 = x.clone().detach()
        x1 = Linear(x)
        x2 = x1 - subtract
        x3 = mean(x2, dim=1, keepdim=True)
        x4 = logsumexp(x3, dim=1, keepdim=True)  # no-op for size-1 dim
        x5 = GELU(x4)
        out = x5 + x0

    Using algebraic simplification:
      mean_n( (xW + b - s)[m, n] )
        = ( x_m · c + sum(b) - sum(s) ) / N,
      where c_k = sum_n W_{k,n}, N = out_features.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        # Store weight as (in_features, out_features) for efficient x @ W
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.has_bias = bias

        # Parameter to subtract (per-output feature)
        self.subtract = nn.Parameter(torch.randn(out_features))

        # Initialize like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x.cuda() if not x.is_cuda else x
        original_x = x  # detach only affects autograd, not forward values

        row_scalars = fused_row_gelu(
            x,
            self.weight,
            self.bias if self.has_bias else None,
            self.subtract,
        )

        out = residual_add(original_x, row_scalars)
        return out
