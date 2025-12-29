import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_row_dot_gelu_residual_kernel(
    x_ptr,          # [M, N]
    c_ptr,          # [N]
    y_ptr,          # [M, N]
    M, N, O,        # batch, in_features, out_features
    col_sum_bias,   # scalar: sum_j(bias_j - subtract_j) or -sum_j(subtract_j)
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    """
    For each row b:
        # Precompute:
        c_i = sum_j W_{j,i}
        col_sum_bias = sum_j(bias_j - subtract_j)   (or -sum_j(subtract_j) if no bias)

        row_sum[b] = sum_i x[b, i] * c_i + col_sum_bias
        m_b        = row_sum[b] / O
        g_b        = GELU(m_b)    # tanh approximation

        y[b, j]    = x[b, j] + g_b
    """
    row = tl.program_id(0)
    row_mask = row < M

    # Offsets for feature dimension
    offs_n = tl.arange(0, BLOCK_N)

    # 1) Compute row-wise dot: sum_i x[b, i] * c_i
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over feature dimension N in chunks of BLOCK_N
    for n_start in range(0, N, BLOCK_N):
        cols = n_start + offs_n
        col_mask = cols < N
        mask = row_mask & col_mask

        x_vals = tl.load(
            x_ptr + row * stride_xm + cols * stride_xn,
            mask=mask,
            other=0.0,
        )
        c_vals = tl.load(
            c_ptr + cols,
            mask=col_mask,  # c_ptr is 1D, no row dependency
            other=0.0,
        )
        acc += tl.sum(x_vals * c_vals, axis=0)

    # 2) Compute mean over out_features and apply GELU
    # m_b = (acc + col_sum_bias) / O
    mean_val = (acc + col_sum_bias) / O

    # GELU via tanh approximation:
    # gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x3 = mean_val * mean_val * mean_val
    inner = sqrt_2_over_pi * (mean_val + 0.044715 * x3)
    # tanh(inner) = (exp(2*inner) - 1) / (exp(2*inner) + 1)
    exp2x = tl.exp(2.0 * inner)
    t = (exp2x - 1.0) / (exp2x + 1.0)
    gelu_val = 0.5 * mean_val * (1.0 + t)

    # 3) Residual add: y[b, j] = x[b, j] + gelu_val (broadcast over features)
    for n_start in range(0, N, BLOCK_N):
        cols = n_start + offs_n
        col_mask = cols < N
        mask = row_mask & col_mask

        x_vals = tl.load(
            x_ptr + row * stride_xm + cols * stride_xn,
            mask=mask,
            other=0.0,
        )
        y_vals = x_vals + gelu_val
        tl.store(
            y_ptr + row * stride_ym + cols * stride_yn,
            y_vals,
            mask=mask,
        )


def fused_linear_sub_avg_logsumexp_gelu_residual(x, weight, bias, subtract):
    """
    Fuses the following sequence:

        y = x @ weight.T + bias          # Linear (Gemm)
        y = y - subtract                 # Subtract
        y = mean(y, dim=1, keepdim=True) # GlobalAvgPool over features
        y = logsumexp(y, dim=1, keepdim=True)  # over a single element: identity
        y = GELU(y)                      # GELU
        out = y + x                      # ResidualAdd (broadcast over features)

    Using the identity:

        y[b, j] = sum_i x[b, i] * W[j, i] + bias[j]
        m_b = (1/O) * sum_j (y[b, j] - subtract[j])
            = (1/O) * (sum_i x[b, i] * sum_j W[j, i] + sum_j (bias[j] - subtract[j]))

    So we precompute:
        c_i          = sum_j W[j, i]
        col_sum_bias = sum_j (bias[j] - subtract[j])   or  -sum_j(subtract[j]) if bias is None

    Then:
        m_b    = (x[b] · c + col_sum_bias) / O
        g_b    = GELU(m_b)
        out[b] = x[b] + g_b
    """
    assert x.is_cuda and weight.is_cuda and subtract.is_cuda
    assert x.dtype == weight.dtype == subtract.dtype

    M, N = x.shape                       # batch, in_features
    O, N_w = weight.shape                # out_features, in_features
    assert N == N_w, "in_features mismatch between input and weight"

    # Precompute column sums of weight: c_i = sum_j W[j, i]
    # Shape: [in_features]
    c = weight.sum(dim=0).contiguous()

    # Precompute scalar bias term: sum_j(bias_j - subtract_j)
    if bias is not None:
        assert bias.is_cuda and bias.dtype == x.dtype
        assert bias.shape == subtract.shape
        col_sum_bias_tensor = (bias - subtract).sum()
    else:
        col_sum_bias_tensor = (-subtract).sum()
    col_sum_bias = float(col_sum_bias_tensor.item())

    y = torch.empty_like(x)

    grid = lambda META: (triton.cdiv(M, 1),)
    fused_row_dot_gelu_residual_kernel[grid](
        x, c, y,
        M, N, O,
        col_sum_bias,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_N=256,
        num_warps=4,
    )
    return y


class ModelNew(nn.Module):
    """
    High-performance Triton implementation of:

        Gemm -> Subtract -> GlobalAvgPool -> LogSumExp -> GELU -> ResidualAdd

    The heavy Gemm + subsequent reductions are analytically simplified and
    implemented by a single row-wise Triton kernel.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Match nn.Linear parameterization: weight shape [out_features, in_features]
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
            self.has_bias = True
        else:
            self.register_buffer("bias", torch.zeros(out_features))
            self.has_bias = False

        # Subtract parameter: shape [out_features]
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # x: [batch, in_features]
        x = x.contiguous()
        bias = self.bias if self.has_bias else None
        return fused_linear_sub_avg_logsumexp_gelu_residual(
            x, self.weight, bias, self.subtract
        )
