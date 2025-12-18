import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=8),
    ],
    key=['H_out', 'W_out'],
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,          # *f32, input:  (N, C, H_in, W_in)
    w_ptr,          # *f32, weight: (C, 1, K, K)
    y_ptr,          # *f32, output: (N, C, H_out, W_out)
    N, C,
    H_in, W_in,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wk1, stride_wk2,
    stride_yn, stride_yc, stride_yh, stride_yw,
    K: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Each program computes up to BLOCK_HW output elements for a single (n, c) pair.
    pid_nc = tl.program_id(0)  # over N*C
    pid_blk = tl.program_id(1)  # tiles over H_out*W_out

    offs_hw = pid_blk * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = offs_hw < (H_out * W_out)

    n = pid_nc // C
    c = pid_nc % C

    # Compute (h_out, w_out) from flattened hw index
    h_out = offs_hw // W_out
    w_out = offs_hw % W_out

    # Base pointers for the current (n, c) slice
    x_base = x_ptr + n * stride_xn + c * stride_xc
    y_base = y_ptr + n * stride_yn + c * stride_yc

    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    # Depthwise convolution: each (n, c) uses its own KxK kernel
    # Weight index for channel c is w[c, 0, kh, kw]
    for kh in range(K):
        for kw in range(K):
            h_in = h_out * stride_h - pad_h + kh
            w_in = w_out * stride_w - pad_w + kw

            in_bounds = (
                (h_in >= 0) & (h_in < H_in) &
                (w_in >= 0) & (w_in < W_in) &
                hw_mask
            )

            x_ptrs = x_base + h_in * stride_xh + w_in * stride_xw
            x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)

            w_idx = c * stride_wn + kh * stride_wk1 + kw * stride_wk2
            w_val = tl.load(w_ptr + w_idx)

            acc += x_vals * w_val

    # Store results
    y_ptrs = y_base + h_out * stride_yh + w_out * stride_yw
    tl.store(y_ptrs, acc, mask=hw_mask)


def depthwise_conv2d_triton(x: torch.Tensor, weight: torch.Tensor, stride: int, padding: int):
    """
    Depthwise 2D convolution (groups = channels) implemented in Triton.
    Only used for float32 CUDA tensors; otherwise falls back to PyTorch conv.
    """
    assert x.dim() == 4, "Input must be NCHW"
    N, C, H_in, W_in = x.shape
    K = weight.shape[-1]
    groups = C
    # Fallback for non-fp32 or non-CUDA tensors
    if (x.device.type != "cuda") or (x.dtype != torch.float32) or (weight.dtype != torch.float32):
        return torch.nn.functional.conv2d(
            x, weight, bias=None, stride=stride, padding=padding, groups=groups
        )

    pad_h = pad_w = padding
    stride_h = stride_w = stride

    H_out = (H_in + 2 * pad_h - K) // stride_h + 1
    W_out = (W_in + 2 * pad_w - K) // stride_w + 1

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    y_contig = y  # already contiguous

    stride_xn, stride_xc, stride_xh, stride_xw = x_contig.stride()
    stride_wn, stride_w1, stride_wk1, stride_wk2 = w_contig.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y_contig.stride()

    grid = lambda META: (
        N * C,
        triton.cdiv(H_out * W_out, META['BLOCK_HW']),
    )

    depthwise_conv2d_kernel[grid](
        x_contig, w_contig, y_contig,
        N, C,
        H_in, W_in,
        H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wn, stride_wk1, stride_wk2,
        stride_yn, stride_yc, stride_yh, stride_yw,
        K=K,
    )
    return y_contig


class ModelNew(nn.Module):
    """
    MBConv block with Triton-accelerated depthwise convolution.

    Fuses only the depthwise conv into Triton for a seed implementation;
    all BatchNorm and ReLU6 operations remain as standard PyTorch ops to
    preserve training semantics.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(ModelNew, self).__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        self.hidden_dim = hidden_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        # Expansion phase (1x1 conv + BN + ReLU6), if expand_ratio != 1
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )

        # Depthwise convolution parameters (groups = hidden_dim)
        self.depthwise_conv_weight = nn.Parameter(
            torch.empty(
                hidden_dim,
                1,
                kernel_size,
                kernel_size,
            )
        )
        nn.init.kaiming_normal_(self.depthwise_conv_weight, mode='fan_out', nonlinearity='relu')
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        self.depthwise_relu = nn.ReLU6(inplace=True)

        # Projection (1x1 conv + BN)
        self.project_conv = nn.Conv2d(
            hidden_dim,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        # Expansion
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)

        # Triton depthwise convolution
        x = depthwise_conv2d_triton(
            x,
            self.depthwise_conv_weight,
            stride=self.stride,
            padding=self.padding,
        )
        x = self.depthwise_bn(x)
        x = self.depthwise_relu(x)

        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)

        # Residual connection
        if self.use_residual:
            x = x + identity

        return x
