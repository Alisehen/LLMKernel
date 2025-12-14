import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=16, num_stages=4),
    ],
    key=['n_elements_per_sample'],
)
@triton.jit
def layer_norm_forward_partial_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sum_sq_ptr,
    n_elements_per_sample,
    sample_stride,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Stage 1: Compute partial sums for each feature block.
    Grid: (num_samples, num_blocks_per_sample) - 2D for maximum parallelism
    """
    pid_sample = tl.program_id(axis=0)
    pid_block = tl.program_id(axis=1)
    
    # Calculate feature block boundaries
    block_start = pid_block * BLOCK_SIZE_N
    offsets = block_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < n_elements_per_sample
    
    # Load data block
    x_ptr_sample = x_ptr + pid_sample * sample_stride
    x_block = tl.load(x_ptr_sample + offsets, mask=mask, other=0.0)
    
    # Compute partial sums
    partial_sum = tl.sum(x_block, axis=0)
    partial_sum_sq = tl.sum(x_block * x_block, axis=0)
    
    # Store partial results
    partial_idx = pid_sample * tl.cdiv(n_elements_per_sample, BLOCK_SIZE_N) + pid_block
    tl.store(partial_sum_ptr + partial_idx, partial_sum)
    tl.store(partial_sum_sq_ptr + partial_idx, partial_sum_sq)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 1024}, num_warps=8, num_stages=4),
    ],
    key=['n_elements_per_sample'],
)
@triton.jit
def layer_norm_forward_final_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    partial_sum_ptr,
    partial_sum_sq_ptr,
    n_elements_per_sample,
    sample_stride,
    weight_bias_stride,
    num_blocks_per_sample,
    eps,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Stage 2: Reduce partial sums, compute statistics, and apply normalization.
    Grid: (num_samples,) - one block per sample for final computation
    """
    pid = tl.program_id(axis=0)
    
    # Pointers for this sample
    x_sample_ptr = x_ptr + pid * sample_stride
    output_sample_ptr = output_ptr + pid * sample_stride
    partial_base = pid * num_blocks_per_sample
    
    # Reduce partial sums with loop unrolling
    sum_acc = 0.0
    sum_sq_acc = 0.0
    count_acc = 0.0
    
    # Unroll loop for better performance
    for block_idx in range(0, num_blocks_per_sample, 4):
        if block_idx + 3 < num_blocks_per_sample:
            # Process 4 blocks at once
            idx0 = partial_base + block_idx
            idx1 = idx0 + 1
            idx2 = idx0 + 2
            idx3 = idx0 + 3
            
            sum0 = tl.load(partial_sum_ptr + idx0)
            sum1 = tl.load(partial_sum_ptr + idx1)
            sum2 = tl.load(partial_sum_ptr + idx2)
            sum3 = tl.load(partial_sum_ptr + idx3)
            
            sum_sq0 = tl.load(partial_sum_sq_ptr + idx0)
            sum_sq1 = tl.load(partial_sum_sq_ptr + idx1)
            sum_sq2 = tl.load(partial_sum_sq_ptr + idx2)
            sum_sq3 = tl.load(partial_sum_sq_ptr + idx3)
            
            sum_acc += (sum0 + sum1 + sum2 + sum3)
            sum_sq_acc += (sum_sq0 + sum_sq1 + sum_sq2 + sum_sq3)
            count_acc += 4.0 * BLOCK_SIZE_M
        else:
            # Process remaining blocks
            for i in range(block_idx, min(block_idx + 4, num_blocks_per_sample)):
                partial_idx = partial_base + i
                sum_acc += tl.load(partial_sum_ptr + partial_idx)
                sum_sq_acc += tl.load(partial_sum_sq_ptr + partial_idx)
                if i == num_blocks_per_sample - 1:
                    last_block_size = n_elements_per_sample - (num_blocks_per_sample - 1) * BLOCK_SIZE_M
                    count_acc += last_block_size
                else:
                    count_acc += BLOCK_SIZE_M
    
    # Compute final statistics
    mean_val = sum_acc / count_acc
    variance = (sum_sq_acc / count_acc) - (mean_val * mean_val)
    rstd_val = 1.0 / tl.sqrt(variance + eps)
    
    # Store statistics
    tl.store(mean_ptr + pid, mean_val)
    tl.store(rstd_ptr + pid, rstd_val)
    
    # Apply normalization with vectorized loads/stores
    for offset in range(0, n_elements_per_sample, BLOCK_SIZE_M):
        offsets = offset + tl.arange(0, BLOCK_SIZE_M)
        mask = offsets < n_elements_per_sample
        
        x_block = tl.load(x_sample_ptr + offsets, mask=mask, other=0.0)
        weight_block = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        bias_block = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        normalized = (x_block - mean_val) * rstd_val
        output_block = normalized * weight_block + bias_block
        
        tl.store(output_sample_ptr + offsets, output_block, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE_B': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_B': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_B': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE_B': 1024}, num_warps=16, num_stages=4),
    ],
    key=['n_elements_per_sample'],
)
@triton.jit
def layer_norm_backward_kernel(
    x_ptr,
    dy_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    dweight_ptr,
    dbias_ptr,
    n_elements_per_sample,
    sample_stride,
    weight_bias_stride,
    BLOCK_SIZE_B: tl.constexpr,
):
    """
    Optimized backward kernel with 2D grid for better parallelism.
    Grid: (num_samples, num_blocks_per_sample)
    """
    pid_sample = tl.program_id(axis=0)
    pid_block = tl.program_id(axis=1)
    
    # Load sample statistics
    mean_val = tl.load(mean_ptr + pid_sample)
    rstd_val = tl.load(rstd_ptr + pid_sample)
    
    # Pointers for this sample and block
    x_sample_ptr = x_ptr + pid_sample * sample_stride
    dy_sample_ptr = dy_ptr + pid_sample * sample_stride
    dx_sample_ptr = dx_ptr + pid_sample * sample_stride
    
    # Calculate block boundaries
    block_start = pid_block * BLOCK_SIZE_B
    offsets = block_start + tl.arange(0, BLOCK_SIZE_B)
    mask = offsets < n_elements_per_sample
    
    # Load data
    x_block = tl.load(x_sample_ptr + offsets, mask=mask, other=0.0)
    dy_block = tl.load(dy_sample_ptr + offsets, mask=mask, other=0.0)
    weight_block = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    
    # Compute normalized input
    x_hat = (x_block - mean_val) * rstd_val
    
    # Local accumulators
    dweight_local = tl.where(mask, dy_block * x_hat, 0.0)
    dbias_local = tl.where(mask, dy_block, 0.0)
    
    # Compute dx components
    dx_block = dy_block * weight_block
    dx_sum_local = tl.sum(dx_block, axis=0)
    dx_dot_local = tl.sum(dx_block * x_hat, axis=0)
    
    # Atomic accumulation for weight and bias gradients
    if dweight_ptr is not None:
        tl.atomic_add(dweight_ptr + offsets, dweight_local, mask=mask)
    if dbias_ptr is not None:
        tl.atomic_add(dbias_ptr + offsets, dbias_local, mask=mask)
    
    # Compute dx directly
    inv_n = 1.0 / n_elements_per_sample
    c1 = dx_sum_local * inv_n * rstd_val
    c2 = dx_dot_local * inv_n * rstd_val
    
    dx_block_final = (dx_block - c1 - x_hat * c2) * rstd_val
    tl.store(dx_sample_ptr + offsets, dx_block_final, mask=mask)

def triton_layer_norm_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """Optimized Triton wrapper for LayerNorm forward pass with 2D grid."""
    original_shape = x.shape
    n_samples = x.shape[0]
    n_elements_per_sample = x.numel() // n_samples
    
    # Reshape to 2D
    x_2d = x.view(n_samples, -1)
    output = torch.empty_like(x_2d)
    
    # Statistics for backward pass
    mean = torch.empty(n_samples, dtype=torch.float32, device=x.device)
    rstd = torch.empty(n_samples, dtype=torch.float32, device=x.device)
    
    # Flatten weight and bias
    weight_flat = weight.view(-1)
    bias_flat = bias.view(-1)
    
    # Stage 1: Partial sums
    def get_grid_partial(meta):
        num_blocks_per_sample = triton.cdiv(n_elements_per_sample, meta['BLOCK_SIZE_N'])
        return (n_samples, num_blocks_per_sample)
    
    partial_sum = torch.zeros(n_samples * triton.cdiv(n_elements_per_sample, 1024), 
                              dtype=torch.float32, device=x.device)
    partial_sum_sq = torch.zeros_like(partial_sum)
    
    layer_norm_forward_partial_kernel[get_grid_partial](
        x_2d,
        partial_sum,
        partial_sum_sq,
        n_elements_per_sample,
        x_2d.stride(0),
    )
    
    # Stage 2: Final reduction and normalization
    def get_grid_final(meta):
        return (n_samples,)
    
    layer_norm_forward_final_kernel[get_grid_final](
        x_2d,
        weight_flat,
        bias_flat,
        output,
        mean,
        rstd,
        partial_sum,
        partial_sum_sq,
        n_elements_per_sample,
        x_2d.stride(0),
        weight_flat.stride(0),
        triton.cdiv(n_elements_per_sample, 1024),
        eps,
    )
    
    return output.view(original_shape), mean, rstd

def triton_layer_norm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
):
    """Optimized Triton wrapper for LayerNorm backward pass with 2D grid."""
    n_samples = x.shape[0]
    n_elements_per_sample = x.numel() // n_samples
    
    x_2d = x.view(n_samples, -1)
    dy_2d = dy.view(n_samples, -1)
    weight_flat = weight.view(-1)
    
    # Gradient tensors
    dx = torch.empty_like(x_2d)
    dweight = torch.zeros_like(weight_flat)
    dbias = torch.zeros_like(weight_flat)
    
    # Grid: 2D for maximum parallelism
    def get_grid(meta):
        num_blocks_per_sample = triton.cdiv(n_elements_per_sample, meta['BLOCK_SIZE_B'])
        return (n_samples, num_blocks_per_sample)
    
    layer_norm_backward_kernel[get_grid](
        x_2d,
        dy_2d,
        weight_flat,
        mean,
        rstd,
        dx,
        dweight,
        dbias,
        n_elements_per_sample,
        x_2d.stride(0),
        weight_flat.stride(0),
    )
    
    return dx.view(x.shape), dweight.view(weight.shape), dbias.view(weight.shape)

class ModelNew(nn.Module):
    """
    Optimized LayerNorm using Triton kernels with improved grid layout.
    """
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _, _ = triton_layer_norm_forward(x, self.weight, self.bias, self.eps)
        return output
