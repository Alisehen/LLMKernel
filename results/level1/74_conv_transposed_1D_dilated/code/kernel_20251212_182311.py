import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def conv_transpose1d_kernel(
    x_ptr,                # Input tensor pointer [B, C_in, L_in]
    w_ptr,                # Weight tensor pointer [C_in, C_out, K]
    b_ptr,                # Bias tensor pointer [C_out] (optional)
    out_ptr,              # Output tensor pointer [B, C_out, L_out]
    B,                    # Batch size
    C_in,                 # Input channels
    C_out,                # Output channels
    L_in,                 # Input length
    L_out,                # Output length
    K,                    # Kernel size
    stride,               # Stride
    padding,              # Padding
    dilation,             # Dilation
    has_bias: tl.constexpr,
    # Tile sizes (must be powers of 2)
    BLOCK_C_IN: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_L: tl.constexpr,
    GROUP_C_IN: tl.constexpr,
):
    """Optimized ConvTranspose1d kernel using tensor cores and blocking."""
    
    # Program IDs: 3D grid for batch, output channel blocks, and output position blocks
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    # Offset calculations for output
    c_out_offset = pid_c * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)
    l_out_offset = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    
    # Mask for valid output channels and positions
    c_out_mask = c_out_offset < C_out
    l_out_mask = l_out_offset < L_out
    
    # Initialize output accumulator [BLOCK_C_OUT, BLOCK_L]
    acc = tl.zeros((BLOCK_C_OUT, BLOCK_L), dtype=tl.float32)
    
    # Process input channels in tiles
    for c_in_block in range(0, C_in, GROUP_C_IN):
        c_in_offset = c_in_block + tl.arange(0, GROUP_C_IN)
        c_in_mask = c_in_offset < C_in
        
        # Load weight tile [GROUP_C_IN, BLOCK_C_OUT, K]
        w_ptrs = w_ptr + (
            c_in_offset[:, None, None] * (C_out * K) +
            c_out_offset[None, :, None] * K +
            tl.arange(0, K)[None, None, :]
        )
        w_tile = tl.load(w_ptrs, mask=c_in_mask[:, None, None] & c_out_mask[None, :, None], other=0.0)
        
        # Process each input position in the tile
        for k_idx in range(K):
            # Calculate corresponding input positions for this kernel element
            l_in_pos = (l_out_offset[None, :] - dilation * k_idx + padding) // stride
            in_bounds = (l_in_pos >= 0) & (l_in_pos < L_in) & ((l_out_offset[None, :] - dilation * k_idx + padding) % stride == 0)
            
            # Load input tile [GROUP_C_IN, BLOCK_L]
            x_ptrs = x_ptr + (
                pid_b * (C_in * L_in) +
                c_in_offset[:, None] * L_in +
                l_in_pos[None, :]
            )
            x_tile = tl.load(x_ptrs, mask=c_in_mask[:, None] & in_bounds, other=0.0)
            
            # Accumulate: x_tile [GROUP_C_IN, BLOCK_L] * w_tile [GROUP_C_IN, BLOCK_C_OUT, k_idx]
            w_slice = w_tile[:, :, k_idx]
            acc += tl.dot(w_slice.T, x_tile, allow_tf32=True)
    
    # Add bias if present
    if has_bias:
        b_ptrs = b_ptr + c_out_offset
        bias = tl.load(b_ptrs, mask=c_out_mask, other=0.0)
        acc += bias[:, None]
    
    # Store output
    out_ptrs = out_ptr + (
        pid_b * (C_out * L_out) +
        c_out_offset[:, None] * L_out +
        l_out_offset[None, :]
    )
    tl.store(out_ptrs, acc, mask=c_out_mask[:, None] & l_out_mask[None, :])


@triton.jit
def conv_transpose1d_bias_kernel(
    output_ptr,
    bias_ptr,
    B,
    C_out,
    L_out,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Separate bias addition kernel for better register usage."""
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    c_out_offset = pid_c * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)
    l_out_offset = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    
    c_out_mask = c_out_offset < C_out
    l_out_mask = l_out_offset < L_out
    
    # Load bias
    bias_ptrs = bias_ptr + c_out_offset
    bias = tl.load(bias_ptrs, mask=c_out_mask, other=0.0)
    
    # Load output, add bias, store back
    out_ptrs = output_ptr + (
        pid_b * (C_out * L_out) +
        c_out_offset[:, None] * L_out +
        l_out_offset[None, :]
    )
    output = tl.load(out_ptrs, mask=c_out_mask[:, None] & l_out_mask[None, :], other=0.0)
    output += bias[:, None]
    tl.store(out_ptrs, output, mask=c_out_mask[:, None] & l_out_mask[None, :])


def triton_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """
    High-performance ConvTranspose1d implementation using Triton.
    
    Args:
        x: Input tensor [B, C_in, L_in]
        weight: Weight tensor [C_in, C_out, K]
        bias: Bias tensor [C_out]
        stride: Stride
        padding: Padding
        dilation: Dilation
    
    Returns:
        Output tensor [B, C_out, L_out]
    """
    B, C_in, L_in = x.shape
    _, C_out, K = weight.shape
    
    # Calculate output length
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    # Allocate output tensor
    output = torch.empty((B, C_out, L_out), device=x.device, dtype=x.dtype)
    
    # Configure kernel parameters
    BLOCK_C_IN = 16  # Input channel tile (power of 2, small for better cache)
    BLOCK_C_OUT = 32  # Output channel tile (power of 2, fits tensor cores)
    BLOCK_L = 64  # Output position tile (power of 2)
    GROUP_C_IN = 16  # Input channel group for dot product
    
    # Adjust tile sizes based on tensor dimensions
    if C_out <= 32:
        BLOCK_C_OUT = 16
    if L_out <= 64:
        BLOCK_L = 32
    
    # Grid configuration
    grid = lambda meta: (
        B,
        triton.cdiv(C_out, meta['BLOCK_C_OUT']),
        triton.cdiv(L_out, meta['BLOCK_L']),
    )
    
    # Launch main kernel (without bias)
    conv_transpose1d_kernel[grid](
        x, weight, 
        bias if bias is not None else torch.empty(0, device=x.device),
        output,
        B, C_in, C_out, L_in, L_out, K,
        stride, padding, dilation,
        has_bias=False,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_L=BLOCK_L,
        GROUP_C_IN=GROUP_C_IN,
    )
    
    # Add bias in separate kernel if present
    if bias is not None:
        conv_transpose1d_bias_kernel[grid](
            output, bias,
            B, C_out, L_out,
            BLOCK_C_OUT=BLOCK_C_OUT,
            BLOCK_L=BLOCK_L,
        )
    
    return output


class ModelNew(nn.Module):
    """
    High-performance ConvTranspose1d implementation using Triton kernels.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size
        stride (int, optional): Stride. Defaults to 1.
        padding (int, optional): Padding. Defaults to 0.
        dilation (int, optional): Dilation. Defaults to 1.
        bias (bool, optional): Whether to use bias. Defaults to False.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Initialize weight with Kaiming uniform
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
        # Initialize bias if requested
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in = in_channels * kernel_size
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Triton kernel."""
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )
