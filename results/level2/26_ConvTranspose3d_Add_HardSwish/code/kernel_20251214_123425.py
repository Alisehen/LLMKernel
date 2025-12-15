import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_add_hardswish_kernel(
    conv_out_ptr,      # [N, C, D, H, W]
    add_input_ptr,     # [N, C, D, H, W]
    out_ptr,           # [N, C, D, H, W]
    total_elements,    # N*C*D*H*W
    N, C, D, H, W,     # Tensor shape dimensions
    stride_conv_n, stride_conv_c, stride_conv_d, stride_conv_h, stride_conv_w,
    stride_add_n, stride_add_c, stride_add_d, stride_add_h, stride_add_w,
    stride_out_n, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: Add + (x * HardSwish(x)) activation
    Optimized for Ada Lovelace with 1D grid covering all output elements
    """
    # 1D grid covering all output elements (N*C*D*H*W)
    pid = tl.program_id(0)
    
    # Compute offsets for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Decompose linear offset into tensor indices using SHAPE dimensions
    linear_idx = offsets
    
    # Use shape dimensions for decomposition (W, H, D, C, N)
    w_idx = linear_idx % W
    tmp = linear_idx // W
    
    h_idx = tmp % H
    tmp = tmp // H
    
    d_idx = tmp % D
    tmp = tmp // D
    
    c_idx = tmp % C
    n_idx = tmp // C
    
    # Compute memory offsets using strides and calculated indices
    conv_offsets = (
        n_idx * stride_conv_n +
        c_idx * stride_conv_c +
        d_idx * stride_conv_d +
        h_idx * stride_conv_h +
        w_idx * stride_conv_w
    )
    
    # Compute memory offsets for add_input
    add_offsets = (
        n_idx * stride_add_n +
        c_idx * stride_add_c +
        d_idx * stride_add_d +
        h_idx * stride_add_h +
        w_idx * stride_add_w
    )
    
    # Compute memory offsets for output
    out_offsets = (
        n_idx * stride_out_n +
        c_idx * stride_out_c +
        d_idx * stride_out_d +
        h_idx * stride_out_h +
        w_idx * stride_out_w
    )
    
    # Load data with mask
    conv_vals = tl.load(conv_out_ptr + conv_offsets, mask=mask, other=0.0)
    add_vals = tl.load(add_input_ptr + add_offsets, mask=mask, other=0.0)
    
    # Add operation: x = conv_out + add_input
    x = conv_vals + add_vals
    
    # Compute HardSwish(x): x * relu6(x + 3) / 6
    x_plus_3 = x + 3.0
    relu6_val = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    
    # x * HardSwish(x) = x * (x * relu6_val / 6)
    # More efficient: x * x * (relu6_val / 6)
    output_val = x * x * (relu6_val * (1.0 / 6.0))
    
    # Store result with mask
    tl.store(out_ptr + out_offsets, output_val, mask=mask)


def fused_add_hardswish(conv_out, add_input):
    """
    Fused Add + (x * HardSwish(x)) operation
    Args:
        conv_out: [N, C, D, H, W] - output from ConvTranspose3d
        add_input: [N, C, D, H, W] - tensor to add (same shape)
    Returns:
        [N, C, D, H, W] - result after Add + (x * HardSwish(x))
    """
    assert conv_out.shape == add_input.shape, "Shapes must match"
    N, C, D, H, W = conv_out.shape
    total_elements = N * C * D * H * W
    
    # Output tensor
    out = torch.empty_like(conv_out)
    
    # Optimized configurations for Ada Lovelace
    configs = [
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=3),
    ]
    
    def grid(meta):
        return (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel with best configuration
    best_config = configs[2]  # 1024 block size, 8 warps
    fused_add_hardswish_kernel[grid](
        conv_out, add_input, out,
        total_elements,
        N, C, D, H, W,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        conv_out.stride(3), conv_out.stride(4),
        add_input.stride(0), add_input.stride(1), add_input.stride(2),
        add_input.stride(3), add_input.stride(4),
        out.stride(0), out.stride(1), out.stride(2),
        out.stride(3), out.stride(4),
        BLOCK_SIZE=best_config.kwargs['BLOCK_SIZE'],
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Fused Add + (x * HardSwish(x)) (Triton)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.register_parameter('bias', nn.Parameter(torch.randn(bias_shape)))

    def forward(self, x, add_input):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).
            add_input (torch.Tensor): Tensor to add after transposed convolution, 
                                     of shape (batch_size, out_channels, D_out, H_out, W_out).
        Returns:
            torch.Tensor: Output tensor after ConvTranspose3d, Add, and x * HardSwish(x).
        """
        # Step 1: PyTorch native ConvTranspose3d
        x = self.conv_transpose(x)
        
        # Step 2: Fused Add + (x * HardSwish(x)) in Triton
        x = fused_add_hardswish(x, add_input)
        
        return x
