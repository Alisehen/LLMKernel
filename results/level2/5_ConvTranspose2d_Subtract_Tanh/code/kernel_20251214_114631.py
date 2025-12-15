# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit
def bias_sub_tanh_kernel(
    x_ptr,        # *f32 / *f16  [N, C, H, W] contiguous (NCHW)
    bias_ptr,     # *f32 / *f16  [C]
    out_ptr,      # *f32 / *f16  [N, C, H, W] contiguous (NCHW)
    NC,           # N * C
    HW,           # H * W
    C,            # channels
    BLOCK_HW: tl.constexpr,
):
    """
    Fused kernel: y = tanh(x - bias)
    Layout: NCHW contiguous
      - We tile over (NC, HW) with a 2D grid:
        pid_nc in [0, NC)
        pid_hw in [0, ceil_div(HW, BLOCK_HW))

    All fused ops share the same linear offset 'offsets' and mask.
    """
    pid_nc = tl.program_id(0)  # over NC
    pid_hw = tl.program_id(1)  # over HW blocks

    # HW indices for this program
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW

    # Linear offsets into the NCHW tensor (contiguous)
    # Tensor is viewed as [NC, HW]
    base = pid_nc * HW
    offsets = base + offs_hw

    # Load x
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index from pid_nc and load bias once per program
    c = pid_nc % C
    bias_val = tl.load(bias_ptr + c)  # scalar, broadcast across BLOCK_HW

    # x - bias
    x_vals = x_vals - bias_val

    # tanh in fp32 for stability: tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    x_fp32 = x_vals.to(tl.float32)
    exp_2x = tl.exp(2.0 * x_fp32)
    tanh_fp32 = (exp_2x - 1.0) / (exp_2x + 1.0)
    tanh_vals = tanh_fp32.to(x_vals.dtype)

    # Store result
    tl.store(out_ptr + offsets, tanh_vals, mask=mask)


def fused_bias_sub_tanh(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused operation:
      y = tanh(x - bias)

    x:    [N, C, H, W], CUDA, contiguous NCHW
    bias: [C, 1, 1] or [C], CUDA
    Returns: in-place modified x (same tensor object).
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert bias.is_cuda, "Bias must be on CUDA device"
    assert x.ndim == 4, "x must be 4D [N, C, H, W]"
    assert x.is_contiguous(), "x must be contiguous NCHW for this optimized kernel"

    N, C, H, W = x.shape
    NC = N * C
    HW = H * W

    # Ensure bias is [C]
    bias_1d = bias.reshape(C)

    # Use x as output (in-place) to avoid extra allocation
    out = x

    # Grid: tile over [NC, HW]
    #   dim0: NC
    #   dim1: ceil_div(HW, BLOCK_HW) (set in autotuned configs)
    def grid(meta):
        return (NC, triton.cdiv(HW, meta["BLOCK_HW"]))

    bias_sub_tanh_kernel[grid](
        x, bias_1d, out,
        NC, HW, C,
    )
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose2d (PyTorch native) + fused (x - bias) + tanh (Triton)
    Optimized for contiguous NCHW on Ada (e.g., RTX 4090).
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape,
                 stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        # PyTorch ConvTranspose2d
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        # Bias parameter with the same shape as the original model
        self.bias = nn.Parameter(torch.randn(*bias_shape))

    def forward(self, x):
        # PyTorch ConvTranspose2d
        x = self.conv_transpose(x)
        # Fused bias subtraction + tanh in Triton
        x = fused_bias_sub_tanh(x, self.bias)
        return x
