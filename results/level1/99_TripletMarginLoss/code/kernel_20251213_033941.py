import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_loss_kernel_2d(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    loss_ptr,
    margin,
    batch_size,
    feature_size,
    stride_batch,
    stride_feat,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_FEAT: tl.constexpr,
    USE_TENSOR_CORES: tl.constexpr,
):
    """
    Optimized 2D kernel for Triplet Margin Loss.
    Each thread block processes a tile of [BLOCK_BATCH x BLOCK_FEAT] elements.
    """
    pid_batch = tl.program_id(axis=0)
    pid_feat = tl.program_id(axis=1)
    
    # Create offsets for the batch dimension
    batch_offsets = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < batch_size
    
    # Create offsets for the feature dimension
    feat_offsets = pid_feat * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
    feat_mask = feat_offsets < feature_size
    
    # Combine masks for 2D tiling
    mask = batch_mask[:, None] & feat_mask[None, :]
    
    # Calculate base pointers for the current tile
    base_anchor = anchor_ptr + batch_offsets[:, None] * stride_batch + feat_offsets[None, :]
    base_positive = positive_ptr + batch_offsets[:, None] * stride_batch + feat_offsets[None, :]
    base_negative = negative_ptr + batch_offsets[:, None] * stride_batch + feat_offsets[None, :]
    
    # Load data in 2D tiles
    if USE_TENSOR_CORES:
        # Use tensor cores for FP16/BF16 inputs with specialized data layout
        anchor = tl.load(base_anchor, mask=mask, other=0.0)
        positive = tl.load(base_positive, mask=mask, other=0.0)
        negative = tl.load(base_negative, mask=mask, other=0.0)
    else:
        anchor = tl.load(base_anchor, mask=mask, other=0.0)
        positive = tl.load(base_positive, mask=mask, other=0.0)
        negative = tl.load(base_negative, mask=mask, other=0.0)
    
    # Compute squared differences
    pos_diff = anchor - positive
    neg_diff = anchor - negative
    pos_sq = pos_diff * pos_diff
    neg_sq = neg_diff * neg_diff
    
    # Allocate shared memory for reduction
    shared_pos = tl.zeros((BLOCK_BATCH, BLOCK_FEAT), dtype=tl.float32)
    shared_neg = tl.zeros((BLOCK_BATCH, BLOCK_FEAT), dtype=tl.float32)
    
    # Accumulate in shared memory (within the tile)
    shared_pos += tl.where(mask, pos_sq, 0.0)
    shared_neg += tl.where(mask, neg_sq, 0.0)
    
    # Reduce across feature dimension within the thread block
    tl.debug_barrier()
    
    # Use tree reduction across features
    reduction_size = BLOCK_FEAT
    while reduction_size > 1:
        half_size = reduction_size // 2
        if feat_offsets[0] < half_size:
            shared_pos[:, feat_offsets[0]] += shared_pos[:, feat_offsets[0] + half_size]
            shared_neg[:, feat_offsets[0]] += shared_neg[:, feat_offsets[0] + half_size]
        reduction_size = half_size
        tl.debug_barrier()
    
    # First thread in each feature group writes to global memory
    if pid_feat == 0:
        # Atomic add to accumulate across feature blocks
        for i in range(BLOCK_BATCH):
            if batch_offsets[i] < batch_size:
                pos_sum_ptr = loss_ptr + batch_offsets[i] * 2  # Store pos and neg separately
                neg_sum_ptr = loss_ptr + batch_offsets[i] * 2 + 1
                tl.atomic_add(pos_sum_ptr, shared_pos[i, 0])
                tl.atomic_add(neg_sum_ptr, shared_neg[i, 0])

@triton.jit
def triplet_finalize_kernel(
    sums_ptr,
    loss_ptr,
    margin,
    batch_size,
    epsilon: tl.constexpr,
):
    """
    Finalize kernel to compute distances and loss from accumulated sums.
    """
    pid = tl.program_id(axis=0)
    
    if pid >= batch_size:
        return
    
    # Load accumulated sums
    pos_sum = tl.load(sums_ptr + pid * 2)
    neg_sum = tl.load(sums_ptr + pid * 2 + 1)
    
    # Compute L2 distances
    pos_dist = tl.sqrt(pos_sum + epsilon)
    neg_dist = tl.sqrt(neg_sum + epsilon)
    
    # Compute final triplet loss
    loss_val = tl.maximum(pos_dist - neg_dist + margin, 0.0)
    tl.store(loss_ptr + pid, loss_val)

def triton_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """
    Compute Triplet Margin Loss using optimized Triton kernels.
    """
    assert anchor.shape == positive.shape == negative.shape
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    
    batch_size, feature_size = anchor.shape
    dtype = anchor.dtype
    
    # Determine if we can use tensor cores
    USE_TENSOR_CORES = dtype in (torch.float16, torch.bfloat16)
    
    # Autotuned configurations for different scenarios
    configs = [
        triton.Config({'BLOCK_BATCH': 64, 'BLOCK_FEAT': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_BATCH': 32, 'BLOCK_FEAT': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_BATCH': 16, 'BLOCK_FEAT': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_BATCH': 128, 'BLOCK_FEAT': 128}, num_warps=8, num_stages=3),
    ]
    
    # Temporary buffer for accumulated sums (pos and neg for each sample)
    sums = torch.zeros(batch_size * 2, device=anchor.device, dtype=torch.float32)
    
    # Allocate output for per-sample losses
    sample_losses = torch.zeros(batch_size, device=anchor.device, dtype=anchor.dtype)
    
    @triton.autotune(configs=configs, key=['batch_size', 'feature_size'])
    def compute_sums(BLOCK_BATCH, BLOCK_FEAT):
        # 2D grid: parallelize across both batch and features
        grid = lambda META: (
            triton.cdiv(batch_size, META['BLOCK_BATCH']),
            triton.cdiv(feature_size, META['BLOCK_FEAT'])
        )
        
        triplet_margin_loss_kernel_2d[grid](
            anchor,
            positive,
            negative,
            sums,
            margin,
            batch_size,
            feature_size,
            anchor.stride(0),
            anchor.stride(1),
            BLOCK_BATCH=BLOCK_BATCH,
            BLOCK_FEAT=BLOCK_FEAT,
            USE_TENSOR_CORES=USE_TENSOR_CORES,
        )
    
    # Launch kernel to compute sums
    compute_sums(batch_size=batch_size, feature_size=feature_size)
    
    # Launch finalization kernel
    grid_final = (triton.cdiv(batch_size, 256),)
    triplet_finalize_kernel[grid_final](
        sums,
        sample_losses,
        margin,
        batch_size,
        epsilon=1e-8,
    )
    
    # Average across batch
    return sample_losses.mean()

class ModelNew(nn.Module):
    """
    A model that computes Triplet Margin Loss using optimized Triton kernels.
    
    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin: float = 1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)
