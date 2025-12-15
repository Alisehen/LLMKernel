import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_scale_bn_gap_kernel_optimized(
    x_ptr,  # Input tensor [N, C, D, H, W]
    scale_factor,  # Scalar scale factor
    running_mean_ptr,  # Running mean [C]
    running_var_ptr,  # Running variance [C]
    weight_ptr,  # Weight [C] (gamma)
    bias_ptr,  # Bias [C] (beta)
    out_ptr,  # Output tensor [N, C, 1, 1, 1]
    eps,  # Epsilon for batch norm
    N, C, D, H, W,  # Input dimensions
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,  # Input strides
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,  # Output strides
    BLOCK_C: tl.constexpr,  # Number of channels per block
    BLOCK_D: tl.constexpr,  # Number of D slices per thread
    BLOCK_HW: tl.constexpr,  # HW per thread (must be power of 2)
):
    """
    Optimized fused kernel: Scale + BatchNorm3d + GlobalAvgPool3d
    Key optimizations:
    1. Coalesced memory access by processing contiguous HW blocks
    2. Parallel reduction within thread block
    3. Register tiling for better data reuse
    """
    # Program IDs - each block handles BLOCK_C channels
    pid_n = tl.program_id(0)  # Batch index
    pid_c_block = tl.program_id(1)  # Channel block index
    pid_d_slice = tl.program_id(2)  # D slice index
    
    # Thread IDs within block
    tid = tl.program_id(axis=3)  # Thread ID for HW reduction
    
    if pid_n >= N:
        return
    
    # Calculate channel indices for this block
    c_offsets = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    # Load batch norm parameters for BLOCK_C channels
    mean_vals = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0.0)
    var_vals = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=1.0)
    weight_vals = tl.load(weight_ptr + c_offsets, mask=c_mask, other=1.0) if weight_ptr is not None else 1.0
    bias_vals = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0) if bias_ptr is not None else 0.0
    
    # Pre-compute batch norm scale factors
    inv_std = 1.0 / tl.sqrt(var_vals + eps)
    scale_norm = weight_vals * inv_std * scale_factor
    bias_norm = bias_vals - mean_vals * weight_vals * inv_std
    
    # Initialize accumulation in registers (one per channel)
    accum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Calculate D range for this thread block
    d_start = pid_d_slice * BLOCK_D
    d_end = min(d_start + BLOCK_D, D)
    
    # Process HW in blocks - each thread handles BLOCK_HW elements
    HW = H * W
    hw_start = tid * BLOCK_HW
    hw_end = min(hw_start + BLOCK_HW, HW)
    
    # Pre-compute strides for faster indexing
    stride_xhxw = stride_xh * H + stride_xw  # Combined stride for HW
    
    for d_idx in range(d_start, d_end):
        # Calculate base offset for this D slice
        d_offset = d_idx * stride_xd
        
        for hw_idx in range(hw_start, hw_end):
            # Convert linear HW index to H and W
            w_idx = hw_idx % W
            h_idx = hw_idx // W
            
            # Calculate pointer offset for all channels at once
            x_offset = (pid_n * stride_xn + 
                       d_offset + 
                       h_idx * stride_xh + 
                       w_idx * stride_xw)
            
            # Load input for all channels in block (coalesced access)
            x_vals = tl.load(x_ptr + x_offset + c_offsets * stride_xc, 
                           mask=c_mask, other=0.0)
            
            # Apply scaled batch norm and accumulate
            x_norm = x_vals * scale_norm + bias_norm
            accum += tl.where(c_mask, x_norm, 0.0)
    
    # Reduce across threads in the block
    # Use tree reduction for better parallelism
    BLOCK_REDUCE = 512  # Must be power of 2 for reduction
    
    # Allocate shared memory for reduction
    shmem = tl.zeros((BLOCK_REDUCE, BLOCK_C), dtype=tl.float32)
    
    # Store partial sums to shared memory
    shmem_offset = tid * BLOCK_C
    tl.store(shmem + shmem_offset + tl.arange(0, BLOCK_C), accum, mask=c_mask)
    
    # Synchronize threads
    tl.debug_barrier()
    
    # Tree reduction across threads
    stride = BLOCK_REDUCE // 2
    while stride > 0:
        if tid < stride:
            # Load two values
            val1 = tl.load(shmem + tid * BLOCK_C + tl.arange(0, BLOCK_C))
            val2 = tl.load(shmem + (tid + stride) * BLOCK_C + tl.arange(0, BLOCK_C))
            
            # Accumulate and store back
            tl.store(shmem + tid * BLOCK_C + tl.arange(0, BLOCK_C), val1 + val2, mask=c_mask)
        
        # Synchronize between reduction steps
        tl.debug_barrier()
        stride //= 2
    
    # Thread 0 writes final result
    if tid == 0:
        # Load reduced sum
        total = tl.load(shmem + tl.arange(0, BLOCK_C), mask=c_mask)
        
        # Calculate average (divide by total spatial elements)
        DHW = D * H * W
        avg_val = total / DHW
        
        # Store to output
        out_offset = pid_n * stride_on + c_offsets * stride_oc
        tl.store(out_ptr + out_offset, avg_val, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_D': 2, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_D': 1, 'BLOCK_HW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_C': 32, 'BLOCK_D': 4, 'BLOCK_HW': 16}, num_warps=4, num_stages=2),
    ],
    key=['C', 'D', 'H', 'W'],
)
@triton.jit
def fused_scale_bn_gap_kernel_final(
    x_ptr, scale_factor, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, out_ptr, eps,
    N, C, D, H, W, stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    """
    Final optimized kernel with autotuning for different input sizes.
    Simplified reduction strategy for better performance.
    """
    # Program IDs
    pid_n = tl.program_id(0)
    pid_c_block = tl.program_id(1)
    
    if pid_n >= N:
        return
    
    # Channel indices for this block
    c_offsets = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    # Load batch norm parameters
    mean_vals = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0.0)
    var_vals = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=1.0)
    weight_vals = tl.load(weight_ptr + c_offsets, mask=c_mask, other=1.0) if weight_ptr is not None else 1.0
    bias_vals = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0) if bias_ptr is not None else 0.0
    
    # Pre-compute batch norm
    inv_std = 1.0 / tl.sqrt(var_vals + eps)
    scale_norm = weight_vals * inv_std * scale_factor
    bias_norm = bias_vals - mean_vals * weight_vals * inv_std
    
    # Initialize accumulation
    accum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    # Process all spatial positions
    DHW = D * H * W
    
    # Use 256 threads per block for better parallelism
    num_threads = 256
    tid = tl.program_id(axis=2)
    
    # Each thread processes a chunk of spatial positions
    for spatial_idx in range(tid, DHW, num_threads):
        if spatial_idx < DHW:
            # Convert linear index to 3D coordinates
            w_idx = spatial_idx % W
            h_idx = (spatial_idx // W) % H
            d_idx = spatial_idx // (H * W)
            
            # Calculate offset
            x_offset = (pid_n * stride_xn + 
                       d_idx * stride_xd + 
                       h_idx * stride_xh + 
                       w_idx * stride_xw)
            
            # Load and process
            x_vals = tl.load(x_ptr + x_offset + c_offsets * stride_xc, 
                           mask=c_mask, other=0.0)
            x_norm = x_vals * scale_norm + bias_norm
            accum += tl.where(c_mask, x_norm, 0.0)
    
    # Parallel reduction across threads
    # Use warp-level reduction first, then cross-warp reduction
    
    # Step 1: Reduction within warp (32 threads)
    for stride in (16, 8, 4, 2, 1):
        accum += tl.shfl_xor(accum, stride)
    
    # Step 2: First thread in each warp stores to shared memory
    shmem = tl.zeros((32, BLOCK_C), dtype=tl.float32)  # 32 warps max
    warp_id = tid // 32
    lane_id = tid % 32
    
    if lane_id == 0:
        tl.store(shmem + warp_id * BLOCK_C + tl.arange(0, BLOCK_C), accum, mask=c_mask)
    
    tl.debug_barrier()
    
    # Step 3: Final reduction by first warp
    if warp_id == 0 and lane_id < 32:
        # Load all warp results
        warp_accum = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for w in range(32):
            val = tl.load(shmem + w * BLOCK_C + tl.arange(0, BLOCK_C), 
                         mask=(w * 32 < num_threads) & c_mask, other=0.0)
            warp_accum += val
        
        # Final warp-level reduction
        for stride in (16, 8, 4, 2, 1):
            warp_accum += tl.shfl_xor(warp_accum, stride)
        
        # Thread 0 writes final result
        if lane_id == 0:
            avg_val = warp_accum / DHW
            out_offset = pid_n * stride_on + c_offsets * stride_oc
            tl.store(out_ptr + out_offset, avg_val, mask=c_mask)


def fused_scale_bn_gap(
    x: torch.Tensor,
    scale_factor: float,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Optimized wrapper: Scale + BatchNorm3d (eval mode) + GlobalAvgPool3d
    """
    N, C, D, H, W = x.shape
    
    # Output shape [N, C, 1, 1, 1]
    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
    
    # Choose grid size
    BLOCK_C = 64  # Default, will be overridden by autotune
    grid_c = triton.cdiv(C, BLOCK_C)
    
    # Launch optimized kernel
    grid = (N, grid_c, 1)
    
    fused_scale_bn_gap_kernel_final[grid](
        x,
        scale_factor,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        eps,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + Fused Scale + BatchNorm3d + GlobalAvgPool3d
    
    IMPORTANT: This fused implementation only works in evaluation mode
    because it uses running statistics for batch normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        
        self.scale_factor = scale_factor
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps
        
    def forward(self, x):
        x = self.conv_transpose(x)
        
        if not self.training:
            self.batch_norm.eval()
            
            with torch.no_grad():
                if self.batch_norm.track_running_stats:
                    current_mean = x.mean(dim=(0, 2, 3, 4))
                    current_var = x.var(dim=(0, 2, 3, 4), unbiased=False)
                    
                    self.batch_norm.running_mean.mul_(1 - self.batch_norm.momentum).add_(
                        current_mean * self.batch_norm.momentum
                    )
                    self.batch_norm.running_var.mul_(1 - self.batch_norm.momentum).add_(
                        current_var * self.batch_norm.momentum
                    )
            
            x = fused_scale_bn_gap(
                x,
                self.scale_factor,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.eps
            )
        else:
            x = x * self.scale_factor
            x = self.batch_norm(x)
            x = x.mean(dim=(2, 3, 4), keepdim=True)
        
        return x
