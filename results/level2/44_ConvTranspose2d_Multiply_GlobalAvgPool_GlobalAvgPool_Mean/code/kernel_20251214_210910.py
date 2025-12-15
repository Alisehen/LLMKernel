import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_multiply_gap_kernel(
    x_ptr,
    out_ptr,
    multiplier,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    USE_SMALL_SPATIAL: tl.constexpr,
):
    """
    Unified kernel optimized for Ada Lovelace with atomic reduction:
    - Atomic addition for spatial reduction across tiles
    - Better grid utilization with 3D launch
    - Aggressive vectorization and loop unrolling
    """
    pid_n = tl.program_id(0)
    pid_c_block = tl.program_id(1)
    pid_hw_block = tl.program_id(2)
    
    # Boundary checks
    if pid_n >= N or pid_c_block * BLOCK_C >= C:
        return
    
    # Channel offsets
    c_offsets = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    spatial_size = H * W
    inv_spatial_size = 1.0 / spatial_size
    
    # Initialize accumulation with proper type
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    if USE_SMALL_SPATIAL:
        # Single tile for small spatial dimensions
        hw_start = 0
        hw_end = spatial_size
    else:
        # Tiled processing for large spatial dimensions
        hw_start = pid_hw_block * BLOCK_HW
        hw_end = min(hw_start + BLOCK_HW, spatial_size)
    
    # Fast path for common case where tile covers entire spatial dimension
    if hw_end - hw_start == BLOCK_HW:
        # Pre-compute all spatial offsets for vectorized loads
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        h = hw_offsets // W
        w = hw_offsets % W
        
        # Base pointer calculation with proper broadcasting
        base_ptr = (
            pid_n * stride_xn + 
            c_offsets[:, None] * stride_xc +
            h[None, :] * stride_xh + 
            w[None, :] * stride_xw
        )
        
        # Load with vectorization hint
        block = tl.load(x_ptr + base_ptr, mask=c_mask[:, None], other=0.0)
        
        # Fast reduction with proper unrolling
        acc += tl.sum(block, axis=1)
    else:
        # Fallback for partial tile
        hw_range = hw_end - hw_start
        # Process in chunks of 32
        for hw_idx in range(0, hw_range, 32):
            hw_offsets = hw_start + hw_idx + tl.arange(0, 32)
            hw_mask = hw_offsets < hw_end
            full_mask = hw_mask & c_mask[:, None]
            
            h = hw_offsets // W
            w = hw_offsets % W
            
            base_ptr = (
                pid_n * stride_xn + 
                c_offsets[:, None] * stride_xc +
                h[None, :] * stride_xh + 
                w[None, :] * stride_xw
            )
            
            block = tl.load(x_ptr + base_ptr, mask=full_mask, other=0.0)
            acc += tl.sum(block, axis=1)
    
    # Fused multiplication and normalization
    result = acc * (multiplier * inv_spatial_size)
    
    # Store results with atomic addition for spatial reduction
    out_base = pid_n * stride_on + c_offsets * stride_oc
    tl.atomic_add(out_ptr + out_base, result, mask=c_mask)


def fused_multiply_gap(x, multiplier):
    """Optimized wrapper with adaptive block sizing and atomic reduction"""
    N, C, H, W = x.shape
    spatial_size = H * W
    
    # Initialize output to zero for atomic reduction
    out = torch.zeros((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    # Adaptive block sizing based on register pressure analysis
    if spatial_size <= 256:
        # Small spatial: maximize occupancy with smaller blocks
        BLOCK_C = min(32, triton.next_power_of_2(C))
        BLOCK_HW = 256
        USE_SMALL_SPATIAL = True
        grid = (N, triton.cdiv(C, BLOCK_C), 1)
    else:
        # Large spatial: balance parallelism and register usage
        BLOCK_C = min(16, triton.next_power_of_2(C))
        BLOCK_HW = 256  # Optimized for memory coalescing
        USE_SMALL_SPATIAL = False
        grid_hw = triton.cdiv(spatial_size, BLOCK_HW)
        grid = (N, triton.cdiv(C, BLOCK_C), min(8, grid_hw))
    
    # Launch kernel
    fused_multiply_gap_kernel[grid](
        x, out, multiplier,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        USE_SMALL_SPATIAL=USE_SMALL_SPATIAL,
        num_warps=4 if BLOCK_C * BLOCK_HW <= 256 else 8,
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose2d (PyTorch native) + Fused multiply + GAP
    with optimized Triton kernel for Ada Lovelace
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, 
            output_padding=output_padding
        )
        self.multiplier = multiplier
    
    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_multiply_gap(x, self.multiplier)
        return x
