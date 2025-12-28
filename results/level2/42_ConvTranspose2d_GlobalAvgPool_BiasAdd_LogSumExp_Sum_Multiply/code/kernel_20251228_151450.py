import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def global_avg_bias_kernel(
    x_ptr, bias_ptr, y_ptr,
    B, C, HW,
    stride_x_n, stride_x_c,
    stride_y_n, stride_y_c,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    bc = pid
    b = bc // C
    c = bc % C

    mask_bc = b < B

    # Base pointer for (b, c, 0, 0)
    base_offset = b * stride_x_n + c * stride_x_c

    acc = 0.0

    # Reduce over spatial HW dimension
    for hw in range(0, HW, BLOCK_HW):
        offs = hw + tl.arange(0, BLOCK_HW)
        mask = (offs < HW) & mask_bc
        ptrs = x_ptr + base_offset + offs
        x_vals = tl.load(ptrs, mask=mask, other=0.0)
        x_vals = tl.astype(x_vals, tl.float32)
        acc += tl.sum(x_vals, axis=0)

    # Global average
    hw_f = tl.float32(HW)
    mean_val = acc / hw_f

    # Add bias (bias is 1D: [C])
    bias_val = tl.load(bias_ptr + c, mask=c < C, other=0.0)
    bias_val = tl.astype(bias_val, tl.float32)

    out_val = mean_val + bias_val

    # Store to y[b, c]
    out_ptr = y_ptr + b * stride_y_n + c * stride_y_c
    tl.store(out_ptr, tl.astype(out_val, tl.float32), mask=mask_bc)


@triton.jit
def logsumexp_mul_kernel(
    y_ptr, out_ptr,
    B, C,
    stride_y_b, stride_y_c,
    stride_o_b,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid
    mask_b = b < B

    # Streaming log-sum-exp over channel dimension
    m = -float("inf")
    s = 0.0

    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask = (offs_c < C) & mask_b
        ptrs = y_ptr + b * stride_y_b + offs_c * stride_y_c
        vals = tl.load(ptrs, mask=mask, other=-float("inf"))
        vals = tl.astype(vals, tl.float32)

        block_max = tl.max(vals, axis=0)
        m_new = tl.maximum(m, block_max)

        s = s * tl.exp(m - m_new)
        s = s + tl.sum(tl.exp(vals - m_new), axis=0)
        m = m_new

    lse = tl.log(s) + m
    out_val = lse * 10.0

    tl.store(out_ptr + b * stride_o_b, tl.astype(out_val, tl.float32), mask=mask_b)


def fused_global_avg_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: [B, C, H, W]
    bias: [C] (flattened from [C,1,1])
    returns y: [B, C] with y[b,c] = mean_{h,w}(x[b,c,h,w]) + bias[c]
    """
    assert x.is_cuda and bias.is_cuda
    B, C, H, W = x.shape
    HW = H * W

    y = torch.empty((B, C), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(B * C, 1),)
    global_avg_bias_kernel[grid](
        x, bias, y,
        B, C, HW,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_HW=256,
    )
    return y


def fused_logsumexp_mul(y: torch.Tensor) -> torch.Tensor:
    """
    y: [B, C]
    returns out: [B, 1] with out[b] = 10 * logsumexp_c(y[b, c])
    """
    assert y.is_cuda
    B, C = y.shape
    out = torch.empty((B, 1), device=y.device, dtype=y.dtype)

    grid = lambda META: (triton.cdiv(B, 1),)
    logsumexp_mul_kernel[grid](
        y, out,
        B, C,
        y.stride(0), y.stride(1),
        out.stride(0),
        BLOCK_C=128,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version:
    - Uses PyTorch ConvTranspose2d (heavy op)
    - Fuses global average pooling + bias add into one Triton kernel
    - Computes log-sum-exp over channels and final *10 in a second Triton kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)                      # [B, C, H_out, W_out]
        bias_1d = self.bias.view(-1)                   # [C]
        x = fused_global_avg_bias(x, bias_1d)          # [B, C]
        x = fused_logsumexp_mul(x)                     # [B, 1]
        return x
