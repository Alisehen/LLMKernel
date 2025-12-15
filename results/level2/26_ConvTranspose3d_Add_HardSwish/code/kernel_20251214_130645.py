import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
    ],
    key=['total_elements']
)
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
    Optimized fused Add + (x * HardSwish(x)) kernel for Ada Lovelace.
    Uses 1D grid with precomputed linear offsets for better memory coalescing.
    """
    # 1D grid covering all output elements
    pid = tl.program_id(0)
    
    # Compute block offsets
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Fast linear index to 5D decomposition using multiplication instead of division
    # Precompute shape products
    CDHW = C * D * H * W
    DHW = D * H * W
    HW = H * W
    
    # Decompose using multiplication and subtraction (faster than division on GPU)
    n_idx = offsets // CDHW
    rest = offsets - n_idx * CDHW
    c_idx = rest // DHW
    rest = rest - c_idx * DHW
    d_idx = rest // HW
    rest = rest - d_idx * HW
    h_idx = rest // W
    w_idx = rest - h_idx * W
    
    # Compute memory offsets using strides (fuse multiplication for efficiency)
    conv_offsets = (
        n_idx * stride_conv_n +
        c_idx * stride_conv_c +
        d_idx * stride_conv_d +
        h_idx * stride_conv_h +
        w_idx * stride_conv_w
    )
    
    add_offsets = (
        n_idx * stride_add_n +
        c_idx * stride_add_c +
        d_idx * stride_add_d +
        h_idx * stride_add_h +
        w_idx * stride_add_w
    )
    
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
    
    # Fused operation: x = conv_out + add_input, then x * HardSwish(x)
    x = conv_vals + add_vals
    
    # CORRECTED: Optimized HardSwish: x * relu6(x + 3) * (1/6)
    x_plus_3 = x + 3.0
    # Fused min(max(x,0),6) using tl.where for better instruction scheduling
    relu6_val = tl.where(x_plus_3 < 0.0, 0.0, tl.where(x_plus_3 > 6.0, 6.0, x_plus_3))
    
    # CORRECTED: x * (relu6_val * 0.16666667) instead of x * x * (relu6_val * 0.16666667)
    output_val = x * (relu6_val * 0.16666667)
    
    # Store result
    tl.store(out_ptr + out_offsets, output_val, mask=mask)


def fused_add_hardswish(conv_out, add_input):
    """
    Fused Add + (x * HardSwish(x)) operation with autotuning
    """
    assert conv_out.shape == add_input.shape, "Shapes must match"
    N, C, D, H, W = conv_out.shape
    total_elements = N * C * D * H * W
    
    # Output tensor
    out = torch.empty_like(conv_out)
    
    # Grid calculation
    def grid(meta):
        return (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel with autotuning
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
