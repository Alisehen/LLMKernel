import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_FEAT': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_FEAT': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_FEAT': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_FEAT': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_FEAT': 256}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE_FEAT': 256}, num_warps=16, num_stages=4),
    ],
    key=['batch_size', 'feature_size']
)
@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    loss_ptr,
    margin,
    batch_size,
    feature_size,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    if pid >= batch_size:
        return
    
    pos_sum_sq = 0.0
    neg_sum_sq = 0.0
    
    base_offset = pid * feature_size
    base_anchor = anchor_ptr + base_offset
    base_positive = positive_ptr + base_offset
    base_negative = negative_ptr + base_offset
    
    # Pre-calculate loop bound
    feature_end = tl.cdiv(feature_size, BLOCK_SIZE_FEAT) * BLOCK_SIZE_FEAT
    
    for feat_start in range(0, feature_end, BLOCK_SIZE_FEAT):
        feat_offsets = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
        mask = feat_offsets < feature_size
        
        anchor = tl.load(base_anchor + feat_offsets, mask=mask, other=0.0)
        positive = tl.load(base_positive + feat_offsets, mask=mask, other=0.0)
        negative = tl.load(base_negative + feat_offsets, mask=mask, other=0.0)
        
        pos_diff = anchor - positive
        neg_diff = anchor - negative
        
        pos_sum_sq += tl.sum(pos_diff * pos_diff, axis=0)
        neg_sum_sq += tl.sum(neg_diff * neg_diff, axis=0)
    
    pos_dist = tl.sqrt(pos_sum_sq + 1e-8)
    neg_dist = tl.sqrt(neg_sum_sq + 1e-8)
    
    loss_val = tl.maximum(pos_dist - neg_dist + margin, 0.0)
    tl.store(loss_ptr + pid, loss_val)

def triton_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    assert anchor.shape == positive.shape == negative.shape
    assert anchor.is_cuda and positive.is_cuda and negative.is_cuda
    
    batch_size, feature_size = anchor.shape
    
    sample_losses = torch.empty(batch_size, device=anchor.device, dtype=anchor.dtype)
    
    grid = (triton.cdiv(batch_size, 1),)
    
    # Pre-calculate for better constant propagation
    max_grid = (batch_size + 1 - 1) // 1
    
    triplet_margin_loss_kernel[grid](
        anchor,
        positive,
        negative,
        sample_losses,
        margin,
        batch_size,
        feature_size,
        BLOCK_SIZE_FEAT=256
    )
    
    return sample_losses.mean()

class ModelNew(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)
