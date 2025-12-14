import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def maxpool1d_forward_kernel(
    x_ptr,
    output_ptr,
    B,
    C,
    L_in,
    L_out,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    """
    Optimized MaxPool1D kernel with vectorized memory access and efficient computation.
    """
    # 2D grid: (C * triton.cdiv(L_out, BLOCK_SIZE_IN), B)
    pid_c_out = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    # Decompose first dimension
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_out = (L_out + BLOCK_SIZE_IN - 1) // BLOCK_SIZE_IN
    pid_c_block = pid_c_out // grid_out
    pid_out_block = pid_c_out % grid_out
    
    # Channel offsets with vectorization
    c_start = pid_c_block * BLOCK_SIZE_C
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    # Output position offsets with vectorization
    out_start = pid_out_block * BLOCK_SIZE_IN
    out_offsets = out_start + tl.arange(0, BLOCK_SIZE_IN)
    out_mask = out_offsets < L_out
    
    # Initialize max values
    max_vals = tl.full((BLOCK_SIZE_C, BLOCK_SIZE_IN), float('-inf'), dtype=tl.float32)
    
    # Process kernel positions
    for k_idx in range(kernel_size):
        # Calculate input positions
        input_positions = out_offsets * stride - padding + k_idx * dilation
        
        # Check input bounds
        input_valid = (input_positions >= 0) & (input_positions < L_in)
        
        # Create 2D mask for loading
        load_mask_2d = out_mask[None, :] & input_valid[None, :] & c_mask[:, None]
        
        # Compute offsets
        batch_offset = pid_b * C * L_in
        offsets_2d = batch_offset + c_offsets[:, None] * L_in + input_positions[None, :]
        
        # Load values
        values = tl.load(
            x_ptr + offsets_2d,
            mask=load_mask_2d,
            other=float('-inf')
        )
        
        # Update max values
        max_vals = tl.maximum(max_vals, values)
    
    # Store results
    store_mask = out_mask[None, :] & c_mask[:, None]
    output_offset_base = pid_b * C * L_out + c_offsets[:, None] * L_out + out_offsets[None, :]
    tl.store(
        output_ptr + output_offset_base,
        max_vals,
        mask=store_mask
    )


def triton_maxpool1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    return_indices: bool = False
) -> torch.Tensor:
    """
    Optimized Max Pooling 1D with efficient kernel selection.
    """
    if stride is None:
        stride = kernel_size
    
    B, C, L_in = x.shape
    
    # Calculate output length (PyTorch formula)
    L_out = (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    # Create output tensor
    output = torch.empty(B, C, L_out, device=x.device, dtype=x.dtype)
    
    # Optimized block size selection
    if L_out >= 128:
        BLOCK_SIZE_IN = 128
    elif L_out >= 64:
        BLOCK_SIZE_IN = 64
    else:
        BLOCK_SIZE_IN = 32
    
    if C >= 64:
        BLOCK_SIZE_C = 64
    elif C >= 32:
        BLOCK_SIZE_C = 32
    else:
        BLOCK_SIZE_C = 16
    
    # Adjust for thread block limits
    total_threads = BLOCK_SIZE_IN * BLOCK_SIZE_C
    if total_threads > 1024:
        scale_factor = (1024 / total_threads) ** 0.5
        BLOCK_SIZE_IN = max(32, int(BLOCK_SIZE_IN * scale_factor))
        BLOCK_SIZE_C = max(16, int(BLOCK_SIZE_C * scale_factor))
    
    # Calculate grid
    grid_c = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_out = (L_out + BLOCK_SIZE_IN - 1) // BLOCK_SIZE_IN
    grid = (grid_c * grid_out, B)
    
    # Determine optimal number of warps
    total_threads = BLOCK_SIZE_IN * BLOCK_SIZE_C
    num_warps = 8 if total_threads >= 512 else 4
    
    # Launch kernel
    maxpool1d_forward_kernel[grid](
        x,
        output,
        B,
        C,
        L_in,
        L_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        BLOCK_SIZE_IN=BLOCK_SIZE_IN,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        num_warps=num_warps,
        num_stages=3
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 1D with Triton kernels.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies optimized Max Pooling 1D to the input tensor.
        """
        return triton_maxpool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=self.return_indices
        )
