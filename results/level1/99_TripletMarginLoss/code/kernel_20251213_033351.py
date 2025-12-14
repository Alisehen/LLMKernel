import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triplet_margin_loss_kernel_optimized(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    loss_ptr,
    margin,
    batch_size,
    feature_size,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    """
    Optimized kernel for Triplet Margin Loss.
    Each thread block processes one sample and reduces across features.
    """
    pid = tl.program_id(axis=0)  # batch index
    
    if pid >= batch_size:
        return
    
    # Initialize accumulators
    pos_sum = tl.zeros((1,), tl.float32)
    neg_sum = tl.zeros((1,), tl.float32)
    
    # Base pointers for current sample
    base_offset = pid * feature_size
    base_anchor = anchor_ptr + base_offset
    base_positive = positive_ptr + base_offset
    base_negative = negative_ptr + base_offset
    
    # Process features in blocks
    for feat_start in range(0, feature_size, BLOCK_SIZE_FEAT):
        # Create offsets for this block
        feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
        mask = feat_offsets < feature_size
        
        # Load data
        anchor = tl.load(base_anchor + feat_offsets, mask=mask, other=0.0)
        positive = tl.load(base_positive + feat_offsets, mask=mask, other=0.0)
        negative = tl.load(base_negative + feat_offsets, mask=mask, other=0.0)
        
        # Compute squared differences
        pos_diff = anchor - positive
        neg_diff = anchor - negative
        pos_sq = pos_diff * pos_diff
        neg_sq = neg_diff * neg_diff
        
        # Accumulate sums
        pos_sum += tl.sum(pos_sq, axis=0)
        neg_sum += tl.sum(neg_sq, axis=0)
    
    # Compute final loss for this sample
    loss_val = tl.maximum(pos_sum - neg_sum + margin, 0.0)
    tl.store(loss_ptr + pid, loss_val)

def triton_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """
    Compute Triplet Margin Loss using optimized Triton kernels.
    
    Args:
        anchor: Tensor of shape (batch_size, feature_dim)
        positive: Tensor of shape (batch_size, feature_dim)
        negative: Tensor of shape (batch_size, feature_dim)
        margin: Margin for the triplet loss
        
    Returns:
        Scalar loss value
    """
    assert anchor.shape == positive.shape == negative.shape
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    
    batch_size, feature_size = anchor.shape
    
    # Allocate output for per-sample losses
    sample_losses = torch.zeros(batch_size, device=anchor.device, dtype=anchor.dtype)
    
    # Configure kernel launch
    BLOCK_SIZE_FEAT = 256  # Optimal for feature dimension
    
    grid = (batch_size,)
    
    # Launch kernel
    triplet_margin_loss_kernel_optimized[grid](
        anchor,
        positive,
        negative,
        sample_losses,
        margin,
        batch_size,
        feature_size,
        BLOCK_SIZE_FEAT=BLOCK_SIZE_FEAT
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
