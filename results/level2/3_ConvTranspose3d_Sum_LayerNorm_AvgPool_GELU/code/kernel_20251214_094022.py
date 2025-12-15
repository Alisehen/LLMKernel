# <complete ModelNew code with optimized Triton kernels>
# <corrected code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_layernorm_lastdim_kernel(
    x_ptr,            # [rows * width]
    ln_weight_ptr,    # [width]
    ln_bias_ptr,      # [width]
    sum_weight_ptr,   # scalar parameter
    out_ptr,          # [rows * width]
    rows,             # number of rows
    width,            # normalization dimension (last dim size)
    inv_width,        # 1.0 / width (float)
    eps,              # layernorm epsilon (float)
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_w = tl.arange(0, BLOCK_W)

    # Compute base index for this row
    row_start = pid * width
    ptrs = x_ptr + row_start + offs_w

    # Masks
    row_mask = pid < rows
    col_mask = offs_w < width
    mask = row_mask & col_mask

    # Load input row
    x = tl.load(ptrs, mask=mask, other=0.0)

    # Load scalar sum_weight and add
    sum_weight = tl.load(sum_weight_ptr)
    x = x + sum_weight

    # Compute mean/var in fp32 for stability
    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=0) * inv_width
    diff = x_f32 - mean
    var = tl.sum(diff * diff, axis=0) * inv_width
    inv_std = tl.rsqrt(var + eps)

    # Load LayerNorm affine parameters
    gamma = tl.load(ln_weight_ptr + offs_w, mask=col_mask, other=1.0)
    beta = tl.load(ln_bias_ptr + offs_w, mask=col_mask, other=0.0)
    gamma = gamma.to(tl.float32)
    beta = beta.to(tl.float32)

    # Normalize and apply affine
    y = diff * inv_std
    y = y * gamma + beta

    # Cast back to input dtype
    y = y.to(x.dtype)

    # Store result
    tl.store(out_ptr + row_start + offs_w, y, mask=mask)


def fused_add_layernorm_lastdim(x: torch.Tensor,
                                sum_weight: torch.Tensor,
                                ln_module: nn.LayerNorm) -> torch.Tensor:
    """
    Fused kernel: x = x + sum_weight; x = LayerNorm(x) over the channel dimension.

    PyTorch reference:
        x = x + sum_weight
        x = nn.LayerNorm(normalized_shape=(C_out,))(x)

    Input:
        x: [N, C_out, D, H, W]

    LayerNorm is defined with normalized_shape=(C_out,), which normalizes over
    the channel dimension (dim=1). For efficient Triton implementation, we
    temporarily move channels to the last dimension and normalize over that.
    """
    assert x.dim() == 5, "Expected 5D input [N, C_out, D, H, W]"
    N, C, D, H, W = x.shape

    # LayerNorm as defined in the original model uses normalized_shape=(out_channels,)
    assert ln_module.normalized_shape == (C,), (
        f"LayerNorm normalized_shape {ln_module.normalized_shape} must match channel dim {C}"
    )

    # Move channels to last dimension so we can normalize over the last dim
    # x_ch_last: [N, D, H, W, C]
    x_ch_last = x.permute(0, 2, 3, 4, 1).contiguous()
    Np, Dp, Hp, Wp, Cp = x_ch_last.shape
    assert (Np, Dp, Hp, Wp, Cp) == (N, D, H, W, C)

    rows = N * D * H * W
    width = C

    x_2d = x_ch_last.view(rows, width)
    out_2d = torch.empty_like(x_2d)

    ln_weight = ln_module.weight
    ln_bias = ln_module.bias
    eps = ln_module.eps

    # Choose BLOCK_W as next power-of-two >= width, up to a reasonable maximum
    MAX_BLOCK_W = 1024
    BLOCK_W = 1 << (width - 1).bit_length()
    if BLOCK_W > MAX_BLOCK_W:
        raise ValueError(
            f"Normalized dimension size {width} exceeds maximum supported BLOCK_W={MAX_BLOCK_W}; "
            f"got width={width}."
        )

    inv_width = 1.0 / float(width)

    grid = lambda META: (rows,)

    # Heuristic for warps: more threads for wider vectors
    num_warps = 4 if BLOCK_W <= 128 else 8

    add_layernorm_lastdim_kernel[grid](
        x_2d,
        ln_weight,
        ln_bias,
        sum_weight,
        out_2d,
        rows,
        width,
        inv_width,
        eps,
        BLOCK_W=BLOCK_W,
        num_warps=num_warps,
    )

    # Restore original layout: [N, D, H, W, C] -> [N, C, D, H, W]
    out_ch_last = out_2d.view(N, D, H, W, C)
    out = out_ch_last.permute(0, 4, 1, 2, 3).contiguous()
    return out


class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a fused
    (add scalar + LayerNorm over channel dim), then average pooling and GELU.

    The fused Triton kernel replaces:
        x = x + self.sum_weight
        x = self.norm(x)

    where self.norm is nn.LayerNorm(normalized_shape=(out_channels,)),
    i.e., normalization is along the channel dimension.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        # Scalar parameter for the sum
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))

        # LayerNorm as in the original model: norm over channel dimension
        self.norm = nn.LayerNorm(norm_shape)

        # Remaining PyTorch ops
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: [N, C_in, D, H, W]
        x = self.conv_transpose(x)  # [N, C_out, D_out, H_out, W_out]

        # Fused: add scalar + LayerNorm over channel dimension
        x = fused_add_layernorm_lastdim(x, self.sum_weight, self.norm)

        x = self.avg_pool(x)
        x = self.gelu(x)
        return x
