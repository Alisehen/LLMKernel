# 1. Imports
import torch
import torch.nn as nn
import triton
import triton.language as tl


# 2. Triton kernel(s)

@triton.autotune(
    configs=[
        # Smaller tile for small W / low register pressure
        triton.Config({'BLOCK_W': 64}, num_warps=2, num_stages=2),
        # Balanced
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
        # Larger tile for big W (fewer loop iterations, better BW)
        triton.Config({'BLOCK_W': 256}, num_warps=4, num_stages=2),
    ],
    key=['H', 'W'],
)
@triton.jit
def gap_bias_kernel(
    x_ptr,          # *f16 / *f32, [N, C, H, W]
    bias_ptr,       # *f16 / *f32, [C]
    y_ptr,          # *same as x, [N, C]
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc,
    inv_hw,         # f32 = 1.0 / (H * W)
    BLOCK_W: tl.constexpr,
):
    """
    y[n, c] = mean_{h,w} x[n, c, h, w] + bias[c]

    Optimized memory pattern:
      - 2D traversal over (H, W) with strictly contiguous accesses along W.
      - No integer division/modulo in the hot loop.
      - All intermediate results kept in registers; single tl.store at the end.
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    if pid_n >= N or pid_c >= C:
        return

    # Base pointer for this (n, c)
    x_base = x_ptr + pid_n * stride_xn + pid_c * stride_xc

    # Vector of column offsets for this program
    offs_w = tl.arange(0, BLOCK_W)
    delta_w = offs_w * stride_xw

    acc = tl.zeros((), dtype=tl.float32)  # scalar accumulator

    row_base = x_base
    h = 0
    while h < H:
        w0 = 0
        # Traverse W dimension in BLOCK_W tiles
        while w0 < W:
            cur_w = w0 + offs_w
            mask = cur_w < W

            x_ptrs = row_base + w0 * stride_xw + delta_w
            x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

            # Reduce this tile into scalar accumulator
            acc += tl.sum(x_vals, axis=0)

            w0 += BLOCK_W
        row_base += stride_xh
        h += 1

    # Global average over H*W
    acc = acc * inv_hw

    # Add bias[c]
    bias_val = tl.load(bias_ptr + pid_c).to(tl.float32)
    acc = acc + bias_val

    # Store result y[n, c]
    y_ptrs = y_ptr + pid_n * stride_yn + pid_c * stride_yc
    tl.store(y_ptrs, acc.to(tl.float32))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_C': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_C': 256}, num_warps=2, num_stages=2),
    ],
    key=['C'],
)
@triton.jit
def logsumexp_mul_kernel(
    y_ptr,      # *f32, [N, C]
    out_ptr,    # *f32, [N, 1]
    N, C,
    stride_yn, stride_yc,
    BLOCK_C: tl.constexpr,
):
    """
    out[n] = 10 * logsumexp_c ( y[n, c] )

    - Two-pass logsumexp for numerical stability.
    - Row-wise reduction over C with BLOCK_C tiling.
    - All intermediates in registers; single tl.store to out.
    """
    pid_n = tl.program_id(0)
    if pid_n >= N:
        return

    row_base = y_ptr + pid_n * stride_yn
    offs_c = tl.arange(0, BLOCK_C)

    # Pass 1: max over C
    max_val = tl.full((), -float('inf'), dtype=tl.float32)
    c0 = 0
    while c0 < C:
        cur_offs = c0 + offs_c
        mask = cur_offs < C
        y_ptrs = row_base + cur_offs * stride_yc
        y_vals = tl.load(y_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
        block_max = tl.max(y_vals, axis=0)
        max_val = tl.maximum(max_val, block_max)
        c0 += BLOCK_C

    # Pass 2: sum(exp(y - max_val))
    sum_exp = tl.zeros((), dtype=tl.float32)
    c0 = 0
    while c0 < C:
        cur_offs = c0 + offs_c
        mask = cur_offs < C
        y_ptrs = row_base + cur_offs * stride_yc
        y_vals = tl.load(y_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
        sum_exp += tl.sum(tl.exp(y_vals - max_val), axis=0)
        c0 += BLOCK_C

    # Final logsumexp and scale by 10.0
    logsumexp_val = max_val + tl.log(sum_exp)
    result = logsumexp_val * 10.0

    tl.store(out_ptr + pid_n, result)


# 3. Wrapper function(s)

def fused_post_convtranspose(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Two-kernel implementation:

      1) gap_bias_kernel:
           y[n, c] = mean(x[n, c, :, :]) + bias[c]   -> y: [N, C]

      2) logsumexp_mul_kernel:
           out[n] = 10 * logsumexp(y[n, :])          -> out: [N, 1]

    Optimizations focusing on memory pattern:
      - gap_bias_kernel performs a 2D (H, W) traversal with strictly contiguous
        accesses along W, eliminating integer div/mod in the hot loop and
        keeping all intermediates in registers (single tl.store for y).
      - logsumexp_mul_kernel keeps a stable, two-pass logsumexp with BLOCK_C
        tiling and only one tl.store to the final output.
    """
    assert x.is_cuda, "Triton kernels require CUDA tensors"
    assert bias.is_cuda, "Bias must be on CUDA device"
    assert x.dim() == 4, "x must be [N, C, H, W]"

    N, C, H, W = x.shape
    assert bias.numel() == C, "Bias must have C elements"

    # Intermediate [N, C] buffer (no intermediates inside kernels themselves)
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)
    out = torch.empty((N, 1), device=x.device, dtype=x.dtype)

    inv_hw = 1.0 / float(H * W)

    # Launch kernel 1: GlobalAvgPool + Bias
    grid_gap = lambda META: (N, C)

    gap_bias_kernel[grid_gap](
        x, bias, y,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1),
        inv_hw,
    )

    # Launch kernel 2: LogSumExp over C + *10.0
    grid_lse = (N,)

    # For numerical stability, compute logsumexp in float32 regardless of x dtype.
    y_f32 = y.to(torch.float32)

    logsumexp_mul_kernel[grid_lse](
        y_f32, out,
        N, C,
        y_f32.stride(0), y_f32.stride(1),
    )

    return out


# 4. ModelNew definition

class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution (PyTorch native),
    followed by optimized Triton kernels for:
      1) GlobalAvgPool over (H, W) + BiasAdd  -> [N, C]
      2) LogSumExp over C + Multiply(10.0)    -> [N, 1]
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        bias_1d = self.bias.view(-1)
        x = fused_post_convtranspose(x, bias_1d)
        return x
