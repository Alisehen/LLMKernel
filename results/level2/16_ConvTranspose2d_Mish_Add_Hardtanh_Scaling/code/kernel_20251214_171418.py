import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_mish_add_hardtanh_scale_kernel_optimized(
    x_ptr,
    out_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_on, stride_oc, stride_oh, stride_ow,
    add_value,
    scale,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel with vectorized memory operations.
    Each thread processes VEC_SIZE elements for better memory throughput.
    """
    # 3D grid: batch, channel_chunks, spatial_chunks
    pid_n = tl.program_id(0)
    pid_c_chunk = tl.program_id(1)
    pid_hw_chunk = tl.program_id(2)
    
    # Thread ID within block (1D block of BLOCK_C * BLOCK_HW threads)
    tid = tl.arange(0, BLOCK_C * BLOCK_HW)
    
    # Decompose thread ID into channel and spatial indices
    c_idx = tid // BLOCK_HW
    hw_base_idx = tid % BLOCK_HW
    
    # Channel offsets for this chunk
    c_offsets = pid_c_chunk * BLOCK_C + c_idx
    c_mask = c_offsets < C
    
    # Spatial indices for VEC_SIZE elements per thread
    hw_idx_base = (pid_hw_chunk * BLOCK_HW + hw_base_idx) * VEC_SIZE
    hw_idx = hw_idx_base + tl.arange(0, VEC_SIZE)
    
    # Compute spatial mask
    hw_mask = hw_idx < (H * W)
    
    # Compute h and w indices
    h_idx = hw_idx // W
    w_idx = hw_idx % W
    
    # Combine masks
    mask = hw_mask[None, :] & c_mask[:, None]
    
    # Compute base pointers
    n_offset = pid_n * stride_xn
    c_offsets_2d = c_offsets[:, None] * stride_xc
    h_offsets = h_idx[None, :] * stride_xh
    w_offsets = w_idx[None, :] * stride_xw
    
    # Load vectorized data
    x_ptrs = x_ptr + n_offset + c_offsets_2d + h_offsets + w_offsets
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # ---------- OPTIMIZED MISH ACTIVATION ----------
    # Numerical stable Mish: x * tanh(softplus(x))
    # Optimized to minimize transcendental ops
    
    # Compute softplus(x) = log(1 + exp(x))
    abs_x = tl.abs(x_vals)
    max_zero_x = tl.maximum(x_vals, 0.0)
    softplus = max_zero_x + tl.log(1.0 + tl.exp(-abs_x))
    
    # Compute tanh(softplus) using optimized formula
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    # We use exp2 = exp(2*softplus) for efficiency
    exp2_softplus = tl.exp(2.0 * softplus)
    tanh_softplus = (exp2_softplus - 1.0) / (exp2_softplus + 1.0 + 1e-8)
    
    # Mish(x) = x * tanh(softplus(x))
    mish_vals = x_vals * tanh_softplus
    
    # Add constant value
    add_vals = mish_vals + add_value
    
    # Hardtanh activation (clamp to [-1, 1])
    # Use min/max for better performance than nested tl.where
    hardtanh_vals = tl.minimum(tl.maximum(add_vals, -1.0), 1.0)
    
    # Scale
    scaled_vals = hardtanh_vals * scale
    
    # Store results
    out_ptrs = out_ptr + pid_n * stride_on + c_offsets_2d + h_offsets + w_offsets
    tl.store(out_ptrs, scaled_vals, mask=mask)


def fused_post_convtranspose(x, add_value, scale):
    """
    Optimized wrapper for fused Mish + Add + Hardtanh + Scale.
    Uses vectorized kernel with automatic tuning.
    """
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    
    # Choose optimal parameters for Ada Lovelace
    BLOCK_C = 16  # Optimal for channel dimension
    BLOCK_HW = 64  # 1024 threads / BLOCK_C = 64
    VEC_SIZE = 4  # Process 4 elements per thread
    
    # Calculate grid dimensions
    channel_chunks = triton.cdiv(C, BLOCK_C)
    spatial_chunks = triton.cdiv(H * W, BLOCK_HW * VEC_SIZE)
    
    grid = (N, channel_chunks, spatial_chunks)
    
    # Launch optimized kernel
    fused_mish_add_hardtanh_scale_kernel_optimized[grid](
        x, out,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        add_value, scale,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        VEC_SIZE=VEC_SIZE,
        num_warps=8,  # Increased for better occupancy
        num_stages=3,  # Pipeline more stages
    )
    
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose2d (PyTorch native) + Optimized Fused Mish + Add + Hardtanh + Scale (Triton)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride, padding, output_padding
        )
        self.add_value = add_value
        self.scale = scale
        
    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose2d
        x = self.conv_transpose(x)
        
        # Step 2: Optimized fused post-ops in Triton
        x = fused_post_convtranspose(x, self.add_value, self.scale)
        return x
