import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BATCH': 64, 'BLOCK_INNER': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_BATCH': 128, 'BLOCK_INNER': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_BATCH': 256, 'BLOCK_INNER': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_BATCH': 32, 'BLOCK_INNER': 512}, num_warps=8, num_stages=3),
    ],
    key=['batch_size', 'inner_dim'],
)
@triton.jit
def kl_div_kernel(
    predictions_ptr,
    targets_ptr,
    output_ptr,
    batch_size,
    inner_dim,
    BLOCK_INNER: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_inner = tl.program_id(axis=1)
    
    batch_start = pid_batch * BLOCK_BATCH
    batch_offsets = batch_start + tl.arange(0, BLOCK_BATCH)
    batch_mask = batch_offsets < batch_size
    
    inner_start = pid_inner * BLOCK_INNER
    inner_offsets = inner_start + tl.arange(0, BLOCK_INNER)
    inner_mask = inner_offsets < inner_dim
    
    batch_idx = batch_offsets[:, None]
    inner_idx = inner_offsets[None, :]
    
    pred_ptrs = predictions_ptr + batch_idx * inner_dim + inner_idx
    target_ptrs = targets_ptr + batch_idx * inner_dim + inner_idx
    
    mask = batch_mask[:, None] & inner_mask[None, :]
    
    pred_block = tl.load(pred_ptrs, mask=mask, other=0.0)
    target_block = tl.load(target_ptrs, mask=mask, other=0.0)
    
    # KL divergence computation matching PyTorch's F.kl_div
    # PyTorch expects: target * (log(target) - input) where input is log(predictions)
    # So we compute: target * (log(target) - log(predictions))
    epsilon = 1e-12
    safe_pred = tl.maximum(pred_block, epsilon)
    safe_target = tl.maximum(target_block, epsilon)
    
    kl_div = safe_target * (tl.log(safe_target) - tl.log(safe_pred))
    
    acc = tl.sum(kl_div, axis=1)
    
    # Atomic add to accumulate results
    output_ptrs = output_ptr + batch_offsets
    tl.atomic_add(output_ptrs, acc, mask=batch_mask)

@triton.jit
def batch_mean_kernel(
    row_sums_ptr,
    output_ptr,
    batch_size,
):
    pid = tl.program_id(axis=0)
    
    # Single block that processes all elements
    offsets = tl.arange(0, batch_size)
    mask = offsets < batch_size
    
    row_sums = tl.load(row_sums_ptr + offsets, mask=mask, other=0.0)
    
    total = tl.sum(row_sums)
    
    # Divide by batch_size for batchmean reduction
    batch_size_float = tl.full((), batch_size, dtype=tl.float32)
    result = total / batch_size_float
    
    # Store single scalar value
    tl.store(output_ptr + pid, result)

def triton_kl_div(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Ensure contiguous memory layout
    predictions = predictions.contiguous()
    targets = targets.contiguous()
    
    # Get tensor dimensions
    batch_size = predictions.shape[0]
    inner_dim = predictions.shape[-1]  # Last dimension for KL divergence
    
    # Allocate output tensors
    row_sums = torch.zeros(batch_size, dtype=torch.float32, device=predictions.device)
    output = torch.zeros(1, dtype=torch.float32, device=predictions.device)
    
    # Launch KL divergence kernel
    grid_batch = triton.cdiv(batch_size, 32)  # Using minimum BLOCK_BATCH from configs
    grid_inner = triton.cdiv(inner_dim, 64)   # Using minimum BLOCK_INNER from configs
    
    kl_div_kernel[(grid_batch, grid_inner)](
        predictions, targets, row_sums,
        batch_size, inner_dim,
    )
    
    # Launch batch mean reduction kernel (single block)
    batch_mean_kernel[(1,)](
        row_sums, output, batch_size,
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        # Apply log to predictions as required by PyTorch's kl_div
        # and ensure both are probabilities (sum to 1)
        predictions = predictions.float()
        targets = targets.float()
        
        return triton_kl_div(predictions.log(), targets)
