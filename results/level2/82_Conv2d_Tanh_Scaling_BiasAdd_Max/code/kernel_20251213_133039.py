import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def fused_activation_kernel_optimized(
    x_ptr,
    bias_ptr,
    output_ptr,
    scaling_factor,
    n_elements,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)
    
    # Vectorized processing
    elements_per_program = tl.cdiv(n_elements, num_pids)
    start = pid * elements_per_program
    end = min(start + elements_per_program, n_elements)
    
    for offset in range(start, end, BLOCK_SIZE * VEC_SIZE):
        vec_offsets = offset + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE + tl.arange(0, VEC_SIZE)[None, :]
        vec_offsets = tl.reshape(vec_offsets, (BLOCK_SIZE * VEC_SIZE,))
        
        mask = vec_offsets < end
        x_vec = tl.load(x_ptr + vec_offsets, mask=mask)
        
        # Compute channel indices with integer arithmetic
        spatial_idx = vec_offsets // HW
        c_idx = spatial_idx % C
        bias_vec = tl.load(bias_ptr + c_idx, mask=mask)
        
        # Fast tanh approximation with better numerical stability
        x2 = 2.0 * x_vec
        exp_neg = tl.exp(-x2)
        tanh_x = 1.0 - 2.0 / (1.0 + exp_neg)
        
        activated = tanh_x * scaling_factor + bias_vec
        tl.store(output_ptr + vec_offsets, activated, mask=mask)


@triton.jit
def max_pool_2d_kernel_optimized(
    input_ptr,
    output_ptr,
    B, C, H, W,
    pool_h, pool_w,
    stride_h, stride_w,
    output_h, output_w,
    BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)
    
    output_size = B * C * output_h * output_w
    positions_per_program = tl.cdiv(output_size, num_pids)
    start_pos = pid * positions_per_program
    end_pos = min(start_pos + positions_per_program, output_size)
    
    # Precompute constants
    output_hw = output_h * output_w
    input_hw = H * W
    
    for pos in range(start_pos, end_pos, BLOCK_K):
        offsets = pos + tl.arange(0, BLOCK_K)
        mask = offsets < end_pos
        
        # Faster index computation using integer division
        w_idx = offsets % output_w
        h_idx = (offsets // output_w) % output_h
        c_idx = (offsets // output_hw) % C
        b_idx = offsets // (output_hw * C)
        
        # Input window with bounds checking
        h_start = h_idx * stride_h
        w_start = w_idx * stride_w
        h_end = min(h_start + pool_h, H)
        w_end = min(w_start + pool_w, W)
        
        # Initialize with vectorized minimum value
        min_val = -3.4028235e38
        max_val = tl.full((BLOCK_K,), min_val, dtype=tl.float32)
        
        # Unrolled pooling loops with early exit
        for ph in range(0, pool_h):
            ph_actual = h_start + ph
            h_in_bounds = (ph_actual < h_end)
            if not tl.reduce(h_in_bounds, "and"):
                break
                
            for pw in range(0, pool_w):
                pw_actual = w_start + pw
                w_in_bounds = (pw_actual < w_end)
                if not tl.reduce(w_in_bounds, "and"):
                    break
                
                # Optimized index calculation
                input_idx = b_idx * C * input_hw + c_idx * input_hw + ph_actual * W + pw_actual
                
                # Load with bounds mask
                load_mask = mask & h_in_bounds & w_in_bounds
                val = tl.load(input_ptr + input_idx, mask=load_mask, other=min_val)
                
                # Vectorized max with conditional
                max_val = tl.where(val > max_val, val, max_val)
        
        # Store with optimized index calculation
        output_idx = b_idx * C * output_hw + c_idx * output_hw + h_idx * output_w + w_idx
        tl.store(output_ptr + output_idx, max_val, mask=mask)


def triton_fused_activation_optimized(x: torch.Tensor, scaling_factor: float, bias: torch.Tensor) -> torch.Tensor:
    if bias.dim() > 1:
        bias = bias.view(-1)
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    B, C, H, W = x.shape
    HW = H * W
    
    # Optimized for memory-bound kernel with vectorization
    VEC_SIZE = 4  # 4-element vectorization for better memory throughput
    BLOCK_SIZE = 256  # Fixed block size for better occupancy
    grid_size = min(65535, triton.cdiv(n_elements, BLOCK_SIZE * VEC_SIZE))
    grid = (grid_size,)
    
    fused_activation_kernel_optimized[grid](
        x, bias, output, scaling_factor, n_elements,
        C, HW,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC_SIZE=VEC_SIZE,
        num_stages=2,  # Add pipelining to hide memory latency
    )
    return output


def triton_max_pool2d_optimized(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    B, C, H, W = x.shape
    stride = kernel_size
    pool_h = pool_w = kernel_size
    
    output_h = H // stride
    output_w = W // stride
    output = torch.empty((B, C, output_h, output_w), device=x.device, dtype=x.dtype)
    
    if output_h <= 0 or output_w <= 0:
        raise ValueError(f"Invalid output dimensions: {output_h}x{output_w}")
    
    # Optimize for low L1 hit rate with larger block size and pipelining
    BLOCK_K = 512  # Increased for better reuse
    output_size = B * C * output_h * output_w
    grid_size = min(65535, triton.cdiv(output_size, BLOCK_K))
    grid = (max(1, grid_size),)
    
    max_pool_2d_kernel_optimized[grid](
        x, output,
        B, C, H, W,
        pool_h, pool_w,
        stride, stride,
        output_h, output_w,
        BLOCK_K=BLOCK_K,
        num_stages=3,  # Increased stages for memory latency hiding
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        
    def forward(self, x):
        x = self.conv(x)
        x = triton_fused_activation_optimized(x, self.scaling_factor, self.bias)
        x = triton_max_pool2d_optimized(x, self.pool_kernel_size)
        return x
