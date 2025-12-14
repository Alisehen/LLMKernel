import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_forward_partial_kernel(
    x_ptr,
    partial_sum_ptr,
    partial_sum_sq_ptr,
    n_elements_per_sample,
    sample_stride,
    BLOCK_SIZE_N: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    """
    Optimized partial kernel with vectorized loads and better memory coalescing.
    Each thread processes VECTOR_SIZE elements to reduce memory transactions.
    """
    pid_sample = tl.program_id(axis=0)
    pid_block = tl.program_id(axis=1)
    tid = tl.program_id(axis=2)
    
    # Calculate feature block boundaries with vectorization
    block_start = pid_block * BLOCK_SIZE_N
    thread_start = block_start + tid * VECTOR_SIZE
    offsets = thread_start + tl.arange(0, VECTOR_SIZE)
    mask = offsets < n_elements_per_sample
    
    # Vectorized load
    x_ptr_sample = x_ptr + pid_sample * sample_stride
    x_vec = tl.load(x_ptr_sample + offsets, mask=mask, other=0.0)
    
    # Compute thread-local sums
    thread_sum = tl.sum(x_vec, axis=0)
    thread_sum_sq = tl.sum(x_vec * x_vec, axis=0)
    thread_count = tl.sum(mask, axis=0)
    
    # Reduce within warp first (warp-level reduction)
    warp_sum = tl.reduce(thread_sum, 0, tl.sum)
    warp_sum_sq = tl.reduce(thread_sum_sq, 0, tl.sum)
    warp_count = tl.reduce(thread_count, 0, tl.sum)
    
    # Only first thread in warp writes result
    if tl.program_id(axis=2) == 0:
        partial_idx = pid_sample * tl.cdiv(n_elements_per_sample, BLOCK_SIZE_N) + pid_block
        tl.store(partial_sum_ptr + partial_idx, warp_sum)
        tl.store(partial_sum_sq_ptr + partial_idx, warp_sum_sq)

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
    BLOCK_SIZE_N: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    """
    Optimized final kernel with better parallelism and memory access patterns.
    Uses warp-level parallelism for reduction and vectorized loads/stores.
    """
    pid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    
    # Pointers for this sample
    x_sample_ptr = x_ptr + pid * sample_stride
    output_sample_ptr = output_ptr + pid * sample_stride
    partial_base = pid * num_blocks_per_sample
    
    # Warp-stride reduction of partial sums
    sum_acc = 0.0
    sum_sq_acc = 0.0
    count_acc = 0.0
    
    # Each warp handles a subset of blocks
    for block_idx in range(tid, num_blocks_per_sample, 32):
        partial_idx = partial_base + block_idx
        sum_acc += tl.load(partial_sum_ptr + partial_idx)
        sum_sq_acc += tl.load(partial_sum_sq_ptr + partial_idx)
        # Compute count for this block
        if block_idx == num_blocks_per_sample - 1:
            last_block_size = n_elements_per_sample - (num_blocks_per_sample - 1) * BLOCK_SIZE_N
            count_acc += last_block_size
        else:
            count_acc += BLOCK_SIZE_N
    
    # Warp reduction to get final sums
    total_sum = tl.reduce(sum_acc, 0, tl.sum)
    total_sum_sq = tl.reduce(sum_sq_acc, 0, tl.sum)
    total_count = tl.reduce(count_acc, 0, tl.sum)
    
    # Compute final statistics (only first thread in warp)
    if tl.program_id(axis=1) == 0:
        mean_val = total_sum / total_count
        variance = (total_sum_sq / total_count) - (mean_val * mean_val)
        rstd_val = 1.0 / tl.sqrt(variance + eps)
        
        tl.store(mean_ptr + pid, mean_val)
        tl.store(rstd_ptr + pid, rstd_val)
    
    # Synchronize to ensure statistics are computed
    tl.debug_barrier()
    
    # Load statistics (broadcast within warp)
    mean_val = tl.load(mean_ptr + pid)
    rstd_val = tl.load(rstd_ptr + pid)
    
    # Apply normalization with vectorized loads/stores
    for offset in range(tid * VECTOR_SIZE, n_elements_per_sample, 32 * VECTOR_SIZE):
        offsets = offset + tl.arange(0, VECTOR_SIZE)
        mask = offsets < n_elements_per_sample
        
        x_block = tl.load(x_sample_ptr + offsets, mask=mask, other=0.0)
        weight_block = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
        bias_block = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
        
        normalized = (x_block - mean_val) * rstd_val
        output_block = normalized * weight_block + bias_block
        
        tl.store(output_sample_ptr + offsets, output_block, mask=mask)

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
    BLOCK_SIZE_N: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
    USE_ATOMICS: tl.constexpr,
):
    """
    Optimized backward kernel with vectorized loads and warp-level reductions.
    Uses atomics for weight/bias gradients when specified.
    """
    pid_sample = tl.program_id(axis=0)
    pid_block = tl.program_id(axis=1)
    tid = tl.program_id(axis=2)
    
    # Load sample statistics
    mean_val = tl.load(mean_ptr + pid_sample)
    rstd_val = tl.load(rstd_ptr + pid_sample)
    
    # Pointers for this sample and block
    x_sample_ptr = x_ptr + pid_sample * sample_stride
    dy_sample_ptr = dy_ptr + pid_sample * sample_stride
    dx_sample_ptr = dx_ptr + pid_sample * sample_stride
    
    # Calculate block boundaries with vectorization
    block_start = pid_block * BLOCK_SIZE_N
    thread_start = block_start + tid * VECTOR_SIZE
    offsets = thread_start + tl.arange(0, VECTOR_SIZE)
    mask = offsets < n_elements_per_sample
    
    # Vectorized loads
    x_vec = tl.load(x_sample_ptr + offsets, mask=mask, other=0.0)
    dy_vec = tl.load(dy_sample_ptr + offsets, mask=mask, other=0.0)
    weight_vec = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    
    # Compute normalized input
    x_hat = (x_vec - mean_val) * rstd_val
    
    # Local accumulators
    dweight_local = tl.where(mask, dy_vec * x_hat, 0.0)
    dbias_local = tl.where(mask, dy_vec, 0.0)
    
    # Compute dx components
    dx_vec = dy_vec * weight_vec
    dx_sum_local = tl.sum(dx_vec, axis=0)
    dx_dot_local = tl.sum(dx_vec * x_hat, axis=0)
    
    # Warp-level reduction for dx components
    dx_sum_warp = tl.reduce(dx_sum_local, 0, tl.sum)
    dx_dot_warp = tl.reduce(dx_dot_local, 0, tl.sum)
    
    # Only first thread in warp accumulates globally
    if tid == 0:
        if USE_ATOMICS:
            # Atomic accumulation for weight and bias gradients
            if dweight_ptr is not None:
                # Reduce dweight_local within warp first
                dweight_warp = tl.reduce(dweight_local, 0, tl.sum)
                tl.atomic_add(dweight_ptr + block_start + pid_block * BLOCK_SIZE_N, dweight_warp)
            if dbias_ptr is not None:
                # Reduce dbias_local within warp first
                dbias_warp = tl.reduce(dbias_local, 0, tl.sum)
                tl.atomic_add(dbias_ptr + block_start + pid_block * BLOCK_SIZE_N, dbias_warp)
    
    # Compute dx with proper normalization
    inv_n = 1.0 / n_elements_per_sample
    c1 = dx_sum_warp * inv_n * rstd_val
    c2 = dx_dot_warp * inv_n * rstd_val
    
    dx_vec_final = (dx_vec - c1 - x_hat * c2) * rstd_val
    tl.store(dx_sample_ptr + offsets, dx_vec_final, mask=mask)

def triton_layer_norm_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """Optimized Triton wrapper for LayerNorm forward pass with improved parallelism."""
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
    
    # Configuration - optimized for Ada Lovelace
    BLOCK_SIZE_N = 1024
    VECTOR_SIZE = 4  # Process 4 elements per thread for better memory coalescing
    BLOCK_THREADS = BLOCK_SIZE_N // VECTOR_SIZE
    
    num_blocks_per_sample = triton.cdiv(n_elements_per_sample, BLOCK_SIZE_N)
    
    # Stage 1: Partial sums with 3D grid for better parallelism
    partial_sum = torch.zeros(n_samples * num_blocks_per_sample, dtype=torch.float32, device=x.device)
    partial_sum_sq = torch.zeros(n_samples * num_blocks_per_sample, dtype=torch.float32, device=x.device)
    
    grid_partial = lambda meta: (n_samples, num_blocks_per_sample, BLOCK_THREADS)
    
    layer_norm_forward_partial_kernel[grid_partial](
        x_2d,
        partial_sum,
        partial_sum_sq,
        n_elements_per_sample,
        x_2d.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        VECTOR_SIZE=VECTOR_SIZE,
        num_stages=3,  # Increased for better latency hiding
    )
    
    # Stage 2: Final reduction and normalization with 2D grid
    grid_final = lambda meta: (n_samples, 32)  # 32 warps per sample
    
    layer_norm_forward_final_kernel[grid_final](
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
        num_blocks_per_sample,
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        VECTOR_SIZE=VECTOR_SIZE,
        num_stages=2,
    )
    
    return output.view(original_shape), mean, rstd

def triton_layer_norm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
):
    """Optimized Triton wrapper for LayerNorm backward pass with vectorized loads."""
    n_samples = x.shape[0]
    n_elements_per_sample = x.numel() // n_samples
    
    x_2d = x.view(n_samples, -1)
    dy_2d = dy.view(n_samples, -1)
    weight_flat = weight.view(-1)
    
    # Gradient tensors
    dx = torch.empty_like(x_2d)
    dweight = torch.zeros_like(weight_flat)
    dbias = torch.zeros_like(weight_flat)
    
    # Configuration
    BLOCK_SIZE_N = 1024
    VECTOR_SIZE = 4
    BLOCK_THREADS = BLOCK_SIZE_N // VECTOR_SIZE
    
    num_blocks_per_sample = triton.cdiv(n_elements_per_sample, BLOCK_SIZE_N)
    
    # Grid: 3D for maximum parallelism
    grid = lambda meta: (n_samples, num_blocks_per_sample, BLOCK_THREADS)
    
    layer_norm_backward_kernel[grid](
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
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        VECTOR_SIZE=VECTOR_SIZE,
        USE_ATOMICS=True,
        num_stages=3,
    )
    
    return dx.view(x.shape), dweight.view(weight.shape), dbias.view(weight.shape)

class ModelNew(nn.Module):
    """
    Optimized LayerNorm using Triton kernels with vectorized loads and improved parallelism.
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
