import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel_3d_grid(
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
):
    # 3D grid: batch, spatial_h, spatial_w
    pid_batch = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Only process if within valid range
    if pid_batch >= batch_size:
        return
    
    # Compute spatial offsets
    h_start = pid_h * BLOCK_SIZE_SPATIAL
    w_start = pid_w * BLOCK_SIZE_SPATIAL
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_SPATIAL)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_SPATIAL)
    
    h_mask = h_offsets < dim_h
    w_mask = w_offsets < dim_w
    spatial_mask = h_mask[:, None] & w_mask[None, :]
    
    # Initialize accumulator for sum of squares
    sum_sq = tl.zeros((BLOCK_SIZE_SPATIAL, BLOCK_SIZE_SPATIAL), dtype=tl.float32)
    
    # Process features in blocks for reduction
    for feat_start in range(0, features, BLOCK_SIZE_FEAT):
        feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
        feat_mask = feat_offsets < features
        
        # Create combined mask
        combined_mask = feat_mask[:, None, None] & spatial_mask[None, :, :]
        
        # Compute pointer offsets
        batch_offset = pid_batch * stride_batch
        feat_offset = feat_offsets[:, None, None] * stride_feat
        h_offset = h_offsets[None, :, None] * stride_h
        w_offset = w_offsets[None, None, :] * stride_w
        
        # Load input block
        block_ptrs = input_ptr + batch_offset + feat_offset + h_offset + w_offset
        x = tl.load(block_ptrs, mask=combined_mask, other=0.0)
        
        # Accumulate sum of squares
        sum_sq += tl.sum(x * x, axis=0)
    
    # Compute RMS: sqrt(mean + eps)
    mean_sq = sum_sq / features
    rms = tl.sqrt(mean_sq + eps)
    
    # Normalize and store in blocks
    for feat_start in range(0, features, BLOCK_SIZE_FEAT):
        feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
        feat_mask = feat_offsets < features
        
        # Create combined mask
        combined_mask = feat_mask[:, None, None] & spatial_mask[None, :, :]
        
        # Compute pointer offsets
        batch_offset = pid_batch * stride_batch
        feat_offset = feat_offsets[:, None, None] * stride_feat
        h_offset = h_offsets[None, :, None] * stride_h
        w_offset = w_offsets[None, None, :] * stride_w
        
        # Load input block
        input_ptrs = input_ptr + batch_offset + feat_offset + h_offset + w_offset
        x = tl.load(input_ptrs, mask=combined_mask, other=0.0)
        
        # Normalize
        normalized = x / rms[None, :, :]
        
        # Store result
        output_ptrs = output_ptr + batch_offset + feat_offset + h_offset + w_offset
        tl.store(output_ptrs, normalized, mask=combined_mask)


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
):
    # Optimized 2D grid: batch×spatial flattened, feature blocks
    pid_spatial_batch = tl.program_id(0)
    pid_feat = tl.program_id(1)
    
    # Decode spatial and batch indices
    spatial_blocks = tl.cdiv(dim_h, BLOCK_SIZE_SPATIAL) * tl.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
    batch_idx = pid_spatial_batch // spatial_blocks
    spatial_flat = pid_spatial_batch % spatial_blocks
    
    # Check bounds
    if batch_idx >= batch_size:
        return
    
    # Decode spatial indices
    spatial_blocks_w = tl.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
    block_h = spatial_flat // spatial_blocks_w
    block_w = spatial_flat % spatial_blocks_w
    
    # Compute spatial offsets with masking
    h_offsets = block_h * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
    w_offsets = block_w * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
    h_mask = h_offsets < dim_h
    w_mask = w_offsets < dim_w
    spatial_mask = h_mask[:, None] & w_mask[None, :]
    
    # Process feature block
    feat_start = pid_feat * BLOCK_SIZE_FEAT
    feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offsets < features
    
    # Check if any feature in this block is valid
    # tl.any doesn't exist, so we use reduction
    if tl.sum(feat_mask) == 0:
        return
    
    # Initialize accumulator for sum of squares
    sum_sq = tl.zeros((BLOCK_SIZE_SPATIAL, BLOCK_SIZE_SPATIAL), dtype=tl.float32)
    
    # Load data and compute sum of squares in a single pass
    for feat_block in range(0, features, BLOCK_SIZE_FEAT):
        # Current feature block offsets
        current_feat_offsets = feat_block + tl.arange(0, BLOCK_SIZE_FEAT)
        current_feat_mask = current_feat_offsets < features
        
        # Create combined mask
        combined_mask = current_feat_mask[:, None, None] & spatial_mask[None, :, :]
        
        # Compute pointer offsets
        batch_offset = batch_idx * stride_batch
        feat_offset = current_feat_offsets[:, None, None] * stride_feat
        h_offset = h_offsets[None, :, None] * stride_h
        w_offset = w_offsets[None, None, :] * stride_w
        
        # Load input block
        block_ptrs = input_ptr + batch_offset + feat_offset + h_offset + w_offset
        x = tl.load(block_ptrs, mask=combined_mask, other=0.0)
        
        # Accumulate sum of squares
        sum_sq += tl.sum(x * x, axis=0)
    
    # Compute RMS: sqrt(mean + eps)
    mean_sq = sum_sq / features
    rms = tl.sqrt(mean_sq + eps)
    
    # Load and normalize current feature block
    combined_mask = feat_mask[:, None, None] & spatial_mask[None, :, :]
    
    batch_offset = batch_idx * stride_batch
    feat_offset = feat_offsets[:, None, None] * stride_feat
    h_offset = h_offsets[None, :, None] * stride_h
    w_offset = w_offsets[None, None, :] * stride_w
    
    input_ptrs = input_ptr + batch_offset + feat_offset + h_offset + w_offset
    x = tl.load(input_ptrs, mask=combined_mask, other=0.0)
    
    # Normalize
    normalized = x / rms[None, :, :]
    
    # Store result
    output_ptrs = output_ptr + batch_offset + feat_offset + h_offset + w_offset
    tl.store(output_ptrs, normalized, mask=combined_mask)


def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    output = torch.empty_like(x)
    
    # Get tensor shape and strides
    batch_size, features, dim_h, dim_w = x.shape
    strides = x.stride()
    
    # Tuned block sizes for 4090
    BLOCK_SIZE_FEAT = 128
    BLOCK_SIZE_SPATIAL = 8
    
    # Choose between kernels based on dimensions
    if batch_size * dim_h * dim_w > 1000000:
        # Use optimized 2D grid for large problems
        grid_spatial = triton.cdiv(dim_h, BLOCK_SIZE_SPATIAL) * triton.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
        grid_batch_spatial = batch_size * grid_spatial
        grid_feat = triton.cdiv(features, BLOCK_SIZE_FEAT)
        
        grid = (grid_batch_spatial, grid_feat)
        kernel = rms_norm_kernel_optimized
    else:
        # Use 3D grid for smaller problems
        grid_batch = batch_size
        grid_h = triton.cdiv(dim_h, BLOCK_SIZE_SPATIAL)
        grid_w = triton.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
        
        grid = (grid_batch, grid_h, grid_w)
        kernel = rms_norm_kernel_3d_grid
    
    # Launch kernel
    kernel[grid](
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
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_rms_norm(x, self.eps)
