import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel_optimized(
    input_ptr,
    output_ptr,
    stride_batch,
    stride_feat,
    stride_h,
    stride_w,
    batch_size,
    features,
    dim_h,
    dim_w,
    eps,
    BLOCK_SIZE_FEAT: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
    NUM_STAGES: tl.constexpr = 2,
):
    # Single-program, multiple-data (SPMD) programming model
    pid = tl.program_id(0)
    
    # Calculate number of spatial blocks
    spatial_blocks_w = tl.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
    spatial_blocks_h = tl.cdiv(dim_h, BLOCK_SIZE_SPATIAL)
    num_spatial_blocks = spatial_blocks_h * spatial_blocks_w
    
    # Decode batch and spatial indices
    batch_idx = pid // num_spatial_blocks
    spatial_idx = pid % num_spatial_blocks
    
    # Decode spatial block coordinates
    block_h = spatial_idx // spatial_blocks_w
    block_w = spatial_idx % spatial_blocks_w
    
    # Check bounds
    if batch_idx >= batch_size:
        return
    
    # Compute spatial offsets
    h_start = block_h * BLOCK_SIZE_SPATIAL
    w_start = block_w * BLOCK_SIZE_SPATIAL
    
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_SPATIAL)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_SPATIAL)
    
    # Create spatial masks
    h_mask = h_offsets < dim_h
    w_mask = w_offsets < dim_w
    spatial_mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize accumulator for sum of squares
    sum_sq = tl.zeros((BLOCK_SIZE_SPATIAL, BLOCK_SIZE_SPATIAL), dtype=tl.float32)
    inv_features = 1.0 / features
    
    # First pass: compute sum of squares across features
    for feat_start in range(0, features, BLOCK_SIZE_FEAT):
        feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
        feat_mask = feat_offsets < features
        
        # Create combined mask
        combined_mask = feat_mask[:, None, None] & spatial_mask[None, :, :]
        
        # Calculate pointer offsets
        batch_offset = batch_idx * stride_batch
        feat_offset = feat_offsets[:, None, None] * stride_feat
        h_offset = h_offsets[None, :, None] * stride_h
        w_offset = w_offsets[None, None, :] * stride_w
        
        # Load input block
        block_ptrs = input_ptr + batch_offset + feat_offset + h_offset + w_offset
        x = tl.load(block_ptrs, mask=combined_mask, other=0.0)
        
        # Accumulate sum of squares
        sum_sq += tl.sum(x * x, axis=0)
    
    # Compute RMS normalization factor
    mean_sq = sum_sq * inv_features
    rms_inv = tl.rsqrt(mean_sq + eps)  # Fast reciprocal square root
    
    # Second pass: apply normalization
    for feat_start in range(0, features, BLOCK_SIZE_FEAT):
        feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
        feat_mask = feat_offsets < features
        
        # Create combined mask
        combined_mask = feat_mask[:, None, None] & spatial_mask[None, :, :]
        
        # Calculate pointer offsets
        batch_offset = batch_idx * stride_batch
        feat_offset = feat_offsets[:, None, None] * stride_feat
        h_offset = h_offsets[None, :, None] * stride_h
        w_offset = w_offsets[None, None, :] * stride_w
        
        # Load input block
        input_ptrs = input_ptr + batch_offset + feat_offset + h_offset + w_offset
        x = tl.load(input_ptrs, mask=combined_mask, other=0.0)
        
        # Apply normalization
        normalized = x * rms_inv[None, :, :]
        
        # Store result
        output_ptrs = output_ptr + batch_offset + feat_offset + h_offset + w_offset
        tl.store(output_ptrs, normalized, mask=combined_mask)


def triton_rms_norm_optimized(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    output = torch.empty_like(x)
    
    # Get tensor shape and strides
    batch_size, features, dim_h, dim_w = x.shape
    strides = x.stride()
    
    # Optimized block sizes for RTX 4090
    BLOCK_SIZE_FEAT = 64  # Reduced for better occupancy
    BLOCK_SIZE_SPATIAL = 16  # Increased for better spatial locality
    
    # Calculate grid
    spatial_blocks_h = triton.cdiv(dim_h, BLOCK_SIZE_SPATIAL)
    spatial_blocks_w = triton.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
    num_spatial_blocks = spatial_blocks_h * spatial_blocks_w
    grid = (batch_size * num_spatial_blocks,)
    
    # Auto-tune num_stages based on tensor size
    tensor_size = batch_size * features * dim_h * dim_w
    num_stages = 3 if tensor_size > 500000 else 2
    
    # Launch kernel
    rms_norm_kernel_optimized[grid](
        x,
        output,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        batch_size,
        features,
        dim_h,
        dim_w,
        eps,
        BLOCK_SIZE_FEAT=BLOCK_SIZE_FEAT,
        BLOCK_SIZE_SPATIAL=BLOCK_SIZE_SPATIAL,
        NUM_STAGES=num_stages,
        num_warps=4,  # Optimized for modern GPUs
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm_optimized(x, self.eps)
