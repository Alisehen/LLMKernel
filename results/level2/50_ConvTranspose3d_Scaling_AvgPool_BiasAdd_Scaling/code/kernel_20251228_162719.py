# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_scale_avgpool3d_bias_scale_kernel(
    x_ptr,          # *const T, input after ConvTranspose3d: [N, C, D_in, H_in, W_in]
    bias_ptr,       # *const T, bias per channel: [C]
    y_ptr,          # *mut T, output after fused ops: [N, C, D_out, H_out, W_out]
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    stride_bias_c,
    scale1,         # float
    scale2,         # float
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    num_out = N * C * D_out * H_out * W_out
    mask = offs < num_out

    # Decode linear index into (n, c, d_out, h_out, w_out)
    tmp = offs
    w_out = tmp % W_out
    tmp = tmp // W_out
    h_out = tmp % H_out
    tmp = tmp // H_out
    d_out = tmp % D_out
    tmp = tmp // D_out
    c = tmp % C
    n = tmp // C

    # Corresponding top-left corner in input for 2x2x2 pool window
    d0 = d_out * 2
    h0 = h_out * 2
    w0 = w_out * 2

    # Base offset in input tensor
    base_offset = (
        n * stride_xn +
        c * stride_xc +
        d0 * stride_xd +
        h0 * stride_xh +
        w0 * stride_xw
    )

    # Compute pointers for 8 elements in the 2x2x2 window
    ptr000 = x_ptr + base_offset
    ptr001 = ptr000 + stride_xw
    ptr010 = ptr000 + stride_xh
    ptr011 = ptr010 + stride_xw
    ptr100 = ptr000 + stride_xd
    ptr101 = ptr100 + stride_xw
    ptr110 = ptr100 + stride_xh
    ptr111 = ptr110 + stride_xw

    v000 = tl.load(ptr000, mask=mask, other=0.0)
    v001 = tl.load(ptr001, mask=mask, other=0.0)
    v010 = tl.load(ptr010, mask=mask, other=0.0)
    v011 = tl.load(ptr011, mask=mask, other=0.0)
    v100 = tl.load(ptr100, mask=mask, other=0.0)
    v101 = tl.load(ptr101, mask=mask, other=0.0)
    v110 = tl.load(ptr110, mask=mask, other=0.0)
    v111 = tl.load(ptr111, mask=mask, other=0.0)

    # Sum over window
    sum_vals = (
        v000 + v001 + v010 + v011 +
        v100 + v101 + v110 + v111
    )

    # avg_pool3d(kernel=2) over x * scale1:
    # mean(x * scale1) = scale1 * (sum / 8)
    inv_pool = 1.0 / 8.0
    pooled = sum_vals * scale1 * inv_pool

    # Load per-channel bias
    bias_val = tl.load(bias_ptr + c * stride_bias_c, mask=mask, other=0.0)

    # Add bias, then final scaling by scale2
    out_val = (pooled + bias_val) * scale2

    # Store to output
    y_offset = (
        n * stride_yn +
        c * stride_yc +
        d_out * stride_yd +
        h_out * stride_yh +
        w_out * stride_yw
    )
    tl.store(y_ptr + y_offset, out_val, mask=mask)


def fused_scale_avgpool3d_bias_scale(x, scale1, bias, scale2):
    """
    Fuses:
      x = x * scale1
      x = AvgPool3d(kernel_size=2, stride=2)(x)
      x = x + bias        # bias: (C, 1, 1, 1)
      x = x * scale2
    into a single Triton kernel.
    """
    assert x.is_cuda, "Input must be on CUDA device"
    x = x.contiguous()
    N, C, D_in, H_in, W_in = x.shape

    # PyTorch AvgPool3d(kernel=2, stride=2) -> output dims floor(in/2)
    D_out = D_in // 2
    H_out = H_in // 2
    W_out = W_in // 2

    y = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Bias is (C, 1, 1, 1) -> flatten to (C,)
    bias_c = bias.view(-1).contiguous()

    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw = x.stride()
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw = y.stride()
    stride_bias_c = bias_c.stride(0)

    num_out = N * C * D_out * H_out * W_out
    grid = lambda META: (triton.cdiv(num_out, META["BLOCK_SIZE"]),)

    fused_scale_avgpool3d_bias_scale_kernel[grid](
        x, bias_c, y,
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
        stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
        stride_bias_c,
        float(scale1),
        float(scale2),
        BLOCK_SIZE=256,
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, then a fused sequence:
      scaling (scale1) + AvgPool3d(kernel=2) + bias add + scaling (scale2)
    implemented with a high-performance Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.scale1 = nn.Parameter(torch.tensor(scale1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scale2 = nn.Parameter(torch.tensor(scale2, dtype=torch.float32))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_scale_avgpool3d_bias_scale(x, self.scale1.item(), self.bias, self.scale2.item())
        return x
