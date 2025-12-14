import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
    ],
    key=['n_elements']
)
@triton.jit
def triplet_margin_loss_kernel(
    anchor_ptr,
    positive_ptr,
    negative_ptr,
    loss_ptr,
    margin,
    n_elements,
    stride_batch,
    stride_feat,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    
    if pid_batch >= stride_batch:
        return
    
    base_offset = pid_batch * stride_feat
    pos_sum_sq = tl.zeros([1], dtype=tl.float32)
    
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load with correct strides
        anchor = tl.load(anchor_ptr + base_offset + offsets, mask=mask, other=0.0)
        positive = tl.load(positive_ptr + base_offset + offsets, mask=mask, other=0.0)
        negative = tl.load(negative_ptr + base_offset + offsets, mask=mask, other=0.0)
        
        pos_diff = anchor - positive
        neg_diff = anchor - negative
        
        pos_sum_sq += tl.sum(pos_diff * pos_diff)
        pos_sum_sq += tl.sum(neg_diff * neg_diff)
    
    pos_dist = tl.sqrt(pos_sum_sq / 2.0 + 1e-8)
    neg_dist = tl.sqrt(pos_sum_sq / 2.0 + 1e-8)
    
    loss_val = tl.maximum(pos_dist - neg_dist + margin, 0.0)
    tl.store(loss_ptr + pid_batch, loss_val)

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
    
    grid = (batch_size,)
    
    triplet_margin_loss_kernel[grid](
        anchor,
        positive,
        negative,
        sample_losses,
        margin,
        feature_size,
        batch_size,
        feature_size,
    )
    
    return sample_losses.mean()

class ModelNew(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ModelNew, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return triton_triplet_margin_loss(anchor, positive, negative, self.margin)
