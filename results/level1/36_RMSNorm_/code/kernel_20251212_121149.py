import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
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
    # Compute spatial block and batch
    pid_spatial = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    # Reconstruct spatial indices
    block_h = pid_spatial // tl.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
    block_w = pid_spatial % tl.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
    
    # Compute offsets
    h_offsets = block_h * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
    w_offsets = block_w * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
    
    # Create masks for spatial dimensions
    h_mask = h_offsets < dim_h
    w_mask = w_offsets < dim_w
    
    # Initialize accumulator for sum of squares
    sum_sq = tl.zeros([BLOCK_SIZE_SPATIAL, BLOCK_SIZE_SPATIAL], dtype=tl.float32)
    
    # Loop over features in blocks
    for feat_start in range(0, features, BLOCK_SIZE_FEAT):
        feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
        feat_mask = feat_offsets < features
        
        # Create combined mask
        mask = feat_mask[:, None, None] & h_mask[None, :] & w_mask[None, None, :]
        
        # Compute pointer offsets for this block
        batch_offset = pid_batch * stride_batch
        feat_offset = feat_offsets[:, None, None] * stride_feat
        h_offset = h_offsets[None, :, None] * stride_h
        w_offset = w_offsets[None, None, :] * stride_w
        
        # Load input block
        block_ptrs = input_ptr + batch_offset + feat_offset + h_offset + w_offset
        x = tl.load(block_ptrs, mask=mask, other=0.0)
        
        # Accumulate sum of squares
        sum_sq += tl.sum(x * x, axis=0)
    
    # Compute RMS: sqrt(mean + eps)
    mean_sq = sum_sq / features
    rms = tl.sqrt(mean_sq + eps)
    
    # Normalize and store
    for feat_start in range(0, features, BLOCK_SIZE_FEAT):
        feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
        feat_mask = feat_offsets < features
        
        # Create combined mask
        mask = feat_mask[:, None, None] & h_mask[None, :] & w_mask[None, None, :]
        
        # Compute pointer offsets for this block
        batch_offset = pid_batch * stride_batch
        feat_offset = feat_offsets[:, None, None] * stride_feat
        h_offset = h_offsets[None, :, None] * stride_h
        w_offset = w_offsets[None, None, :] * stride_w
        
        # Load input block
        block_ptrs = input_ptr + batch_offset + feat_offset + h_offset + w_offset
        x = tl.load(block_ptrs, mask=mask, other=0.0)
        
        # Normalize
        normalized = x / rms[None, :, :]
        
        # Store result
        output_ptrs = output_ptr + batch_offset + feat_offset + h_offset + w_offset
        tl.store(output_ptrs, normalized, mask=mask)


@triton.jit
def rms_norm_kernel_small_features(
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
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    # This kernel is optimized for small feature dimensions (<= 128)
    # We process all features in a single block to avoid the feature loop
    
    pid_spatial = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    # Reconstruct spatial indices
    block_h = pid_spatial // tl.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
    block_w = pid_spatial % tl.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
    
    # Compute offsets
    h_offsets = block_h * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
    w_offsets = block_w * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
    
    # Create masks for spatial dimensions
    h_mask = h_offsets < dim_h
    w_mask = w_offsets < dim_w
    
    # Create feature offsets
    feat_offsets = tl.arange(0, features)
    feat_mask = feat_offsets < features
    
    # Create combined mask
    mask = feat_mask[:, None, None] & h_mask[None, :] & w_mask[None, None, :]
    
    # Compute pointer offsets
    batch_offset = pid_batch * stride_batch
    feat_offset = feat_offsets[:, None, None] * stride_feat
    h_offset = h_offsets[None, :, None] * stride_h
    w_offset = w_offsets[None, None, :] * stride_w
    
    # Load input block
    block_ptrs = input_ptr + batch_offset + feat_offset + h_offset + w_offset
    x = tl.load(block_ptrs, mask=mask, other=0.0)
    
    # Compute sum of squares
    sum_sq = tl.sum(x * x, axis=0)
    
    # Compute RMS
    mean_sq = sum_sq / features
    rms = tl.sqrt(mean_sq + eps)
    
    # Normalize and store
    normalized = x / rms[None, :, :]
    tl.store(block_ptrs, normalized, mask=mask)


def triton_rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    output = torch.empty_like(x)
    
    # Get tensor shape and strides
    batch_size, features, dim_h, dim_w = x.shape
    strides = x.stride()
    
    # Choose kernel based on feature size
    if features <= 128:
        # Use small features kernel
        BLOCK_SIZE_SPATIAL = 16  # 16x16 = 256 elements per block
        grid_spatial = tl.cdiv(dim_h, BLOCK_SIZE_SPATIAL) * tl.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
        grid_batch = batch_size
        
        rms_norm_kernel_small_features[(grid_spatial, grid_batch)](
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
            BLOCK_SIZE_SPATIAL=BLOCK_SIZE_SPATIAL,
        )
    else:
        # Use general kernel with feature blocking
        BLOCK_SIZE_FEAT = 64
        BLOCK_SIZE_SPATIAL = 8  # 8x8 = 64 spatial elements per block
        
        grid_spatial = tl.cdiv(dim_h, BLOCK_SIZE_SPATIAL) * tl.cdiv(dim_w, BLOCK_SIZE_SPATIAL)
        grid_batch = batch_size
        
        rms_norm_kernel[(grid_spatial, grid_batch)](
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
