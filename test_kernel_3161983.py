import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def avg_pool_3d_kernel(
    x_ptr,
    output_ptr,
    # Input dimensions
    B, C, D, H, W,
    # Output dimensions
    out_D, out_H, out_W,
    # Kernel and stride parameters
    kernel_d: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    stride_d: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_d: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    # Tensor strides
    stride_b: tl.constexpr,
    stride_c: tl.constexpr,
    stride_d_in: tl.constexpr,
    stride_h_in: tl.constexpr,
    stride_w_in: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Optimized 3D average pooling kernel.
    Each thread block processes multiple output elements.
    """
    # Program IDs
    pid_bc = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)
    
    # Decompose batch and channel
    batch_id = pid_bc // C
    channel_id = pid_bc % C
    
    # Decompose height and width block
    blocks_w = tl.cdiv(out_W, BLOCK_W)
    block_h_id = pid_hw // blocks_w
    block_w_id = pid_hw % blocks_w
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_D, BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    # Output indices for this block
    out_d_base = pid_d * BLOCK_D
    out_h_base = block_h_id * BLOCK_H
    out_w_base = block_w_id * BLOCK_W
    
    # Create masks for valid output positions
    out_d_idx = out_d_base + tl.arange(0, BLOCK_D)
    out_h_idx = out_h_base + tl.arange(0, BLOCK_H)
    out_w_idx = out_w_base + tl.arange(0, BLOCK_W)
    
    mask_d = out_d_idx < out_D
    mask_h = out_h_idx < out_H
    mask_w = out_w_idx < out_W
    
    mask_output = mask_d[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    
    # Loop over kernel window
    for kd in range(kernel_d):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                # Compute input positions for the entire block
                in_d = out_d_idx * stride_d - pad_d + kd
                in_h = out_h_idx * stride_h - pad_h + kh
                in_w = out_w_idx * stride_w - pad_w + kw
                
                # Create masks for valid input positions
                mask_d_in = (in_d >= 0) & (in_d < D)
                mask_h_in = (in_h >= 0) & (in_h < H)
                mask_w_in = (in_w >= 0) & (in_w < W)
                
                mask_input = mask_d_in[:, None, None] & mask_h_in[None, :, None] & mask_w_in[None, None, :]
                
                # Combined mask - both output and input must be valid
                mask_valid = mask_output & mask_input
                
                # Compute input offsets
                offset = (batch_id * stride_b + 
                         channel_id * stride_c + 
                         in_d[:, None, None] * stride_d_in + 
                         in_h[None, :, None] * stride_h_in + 
                         in_w[None, None, :] * stride_w_in)
                
                # Load input values with mask
                val = tl.load(x_ptr + offset, mask=mask_valid, other=0.0)
                
                # Accumulate
                accumulator += val
    
    # Compute average - divide by total kernel elements
    kernel_size_3d = kernel_d * kernel_h * kernel_w
    avg_val = accumulator / kernel_size_3d
    
    # Store results
    out_offset = (batch_id * C * out_D * out_H * out_W +
                 channel_id * out_D * out_H * out_W +
                 out_d_idx[:, None, None] * out_H * out_W +
                 out_h_idx[None, :, None] * out_W +
                 out_w_idx[None, None, :])
    
    tl.store(output_ptr + out_offset, avg_val, mask=mask_output)

def triton_avg_pool_3d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0
) -> torch.Tensor:
    """
    Triton-optimized 3D average pooling.
    
    Args:
        x: Input tensor of shape (B, C, D, H, W)
        kernel_size: Size of the pooling kernel
        stride: Stride for pooling (defaults to kernel_size)
        padding: Padding applied to input
        
    Returns:
        Pooled tensor
    """
    if x.dim() != 5:
        raise ValueError(f"Input tensor must be 5D, got {x.dim()}D")
    
    if stride is None:
        stride = kernel_size
    
    B, C, D, H, W = x.shape
    
    # Compute output dimensions (PyTorch formula)
    out_D = (D + 2 * padding - kernel_size) // stride + 1
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1
    
    # Allocate output tensor
    output = torch.empty((B, C, out_D, out_H, out_W), device=x.device, dtype=x.dtype)
    
    # Ensure input is contiguous
    x = x.contiguous()
    
    # Compute input tensor strides
    stride_b = C * D * H * W
    stride_c = D * H * W
    stride_d_in = H * W
    stride_h_in = W
    stride_w_in = 1
    
    # Block size configuration - optimized for 3D pooling
    BLOCK_D, BLOCK_H, BLOCK_W = 4, 8, 8
    
    # Grid configuration
    grid_bc = B * C
    grid_d = triton.cdiv(out_D, BLOCK_D)
    grid_h = triton.cdiv(out_H, BLOCK_H)
    grid_w = triton.cdiv(out_W, BLOCK_W)
    grid_hw = grid_h * grid_w
    
    # Launch kernel
    avg_pool_3d_kernel[(grid_bc, grid_d, grid_hw)](
        x,
        output,
        B, C, D, H, W,
        out_D, out_H, out_W,
        kernel_size, kernel_size, kernel_size,
        stride, stride, stride,
        padding, padding, padding,
        stride_b, stride_c, stride_d_in, stride_h_in, stride_w_in,
        BLOCK_D, BLOCK_H, BLOCK_W,
    )
    
    return output

class ModelNew(nn.Module):
    """
    Simple model that performs 3D Average Pooling using Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the kernel to apply pooling.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which uses the kernel size.
            padding (int, optional): Padding to apply before pooling. Defaults to 0.
        """
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Average Pooling to the input tensor using Triton kernels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied, shape depends on kernel_size, stride and padding.
        """
        return triton_avg_pool_3d(x, self.kernel_size, self.stride, self.padding)
