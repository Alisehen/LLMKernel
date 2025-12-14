import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    mean_ptr,
    rstd_ptr,
    n_elements_per_sample,
    sample_stride,
    weight_bias_stride,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Kernel for Layer Normalization forward pass.
    Each program processes one sample (batch dimension).
    """
    # Get sample index
    pid = tl.program_id(axis=0)
    
    # Pointers for this sample
    x_sample_ptr = x_ptr + pid * sample_stride
    output_sample_ptr = output_ptr + pid * sample_stride
    
    # Initialize accumulators for mean and variance
    mean_acc = 0.0
    m2_acc = 0.0
    count_acc = 0.0
    
    # First pass: compute mean and variance using Welford's online algorithm
    for offset in range(0, n_elements_per_sample, BLOCK_SIZE_N):
        # Create offsets for this block
        col_offsets = offset + tl.arange(0, BLOCK_SIZE_N)
        mask = col_offsets < n_elements_per_sample
        
        # Load block of data
        x_block = tl.load(x_sample_ptr + col_offsets, mask=mask, other=0.0)
        
        # Welford's algorithm update
        for i in range(0, BLOCK_SIZE_N, 1):
            idx = tl.where(col_offsets + i < n_elements_per_sample, i, -1)
            if idx >= 0:
                val = tl.load(x_sample_ptr + col_offsets[i:i+1], mask=col_offsets[i:i+1] < n_elements_per_sample)
                # Online mean and variance update
                count_acc += 1.0
                delta = val - mean_acc
                mean_acc += delta / count_acc
                delta2 = val - mean_acc
                m2_acc += delta * delta2
    
    # Compute final variance and rstd
    variance = m2_acc / count_acc
    rstd_val = 1.0 / tl.sqrt(variance + eps)
    
    # Store mean and rstd for backward pass if needed
    if mean_ptr is not None:
        tl.store(mean_ptr + pid, mean_acc)
    if rstd_ptr is not None:
        tl.store(rstd_ptr + pid, rstd_val)
    
    # Second pass: normalize and apply affine transformation
    for offset in range(0, n_elements_per_sample, BLOCK_SIZE_N):
        # Create offsets for this block
        col_offsets = offset + tl.arange(0, BLOCK_SIZE_N)
        mask = col_offsets < n_elements_per_sample
        
        # Load data and parameters
        x_block = tl.load(x_sample_ptr + col_offsets, mask=mask, other=0.0)
        weight_block = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        bias_block = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
        
        # Normalize: (x - mean) * rstd
        normalized = (x_block - mean_acc) * rstd_val
        
        # Apply affine transformation
        output_block = normalized * weight_block + bias_block
        
        # Store result
        tl.store(output_sample_ptr + col_offsets, output_block, mask=mask)

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
):
    """
    Kernel for Layer Normalization backward pass.
    Each program processes one sample (batch dimension).
    """
    pid = tl.program_id(axis=0)
    
    # Pointers for this sample
    x_sample_ptr = x_ptr + pid * sample_stride
    dy_sample_ptr = dy_ptr + pid * sample_stride
    dx_sample_ptr = dx_ptr + pid * sample_stride
    
    # Load mean and rstd for this sample
    mean_val = tl.load(mean_ptr + pid)
    rstd_val = tl.load(rstd_ptr + pid)
    
    # Initialize accumulators for gradients
    dweight_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    dbias_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    dx_sum_acc = 0.0
    dx_dot_acc = 0.0
    
    # First pass: compute gradients and statistics
    for offset in range(0, n_elements_per_sample, BLOCK_SIZE_N):
        # Create offsets for this block
        col_offsets = offset + tl.arange(0, BLOCK_SIZE_N)
        mask = col_offsets < n_elements_per_sample
        
        # Load data
        x_block = tl.load(x_sample_ptr + col_offsets, mask=mask, other=0.0)
        dy_block = tl.load(dy_sample_ptr + col_offsets, mask=mask, other=0.0)
        weight_block = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        
        # Normalized input
        x_hat = (x_block - mean_val) * rstd_val
        
        # Gradients for bias and weight
        dbias_acc += tl.where(mask, dy_block, 0.0)
        dweight_acc += tl.where(mask, dy_block * x_hat, 0.0)
        
        # Gradients for dx (partial)
        dx_block = dy_block * weight_block
        dx_sum_acc += tl.sum(dx_block, axis=0)
        dx_dot_acc += tl.sum(dx_block * x_hat, axis=0)
    
    # Reduction across blocks (in-register reduction already done)
    # Scale factors for dx
    inv_n = 1.0 / n_elements_per_sample
    c1 = dx_sum_acc * inv_n * rstd_val
    c2 = dx_dot_acc * inv_n * rstd_val
    
    # Second pass: compute dx
    for offset in range(0, n_elements_per_sample, BLOCK_SIZE_N):
        # Create offsets for this block
        col_offsets = offset + tl.arange(0, BLOCK_SIZE_N)
        mask = col_offsets < n_elements_per_sample
        
        # Load data
        x_block = tl.load(x_sample_ptr + col_offsets, mask=mask, other=0.0)
        dy_block = tl.load(dy_sample_ptr + col_offsets, mask=mask, other=0.0)
        weight_block = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        
        # Normalized input
        x_hat = (x_block - mean_val) * rstd_val
        
        # Compute dx
        dx_block = dy_block * weight_block
        dx_block = (dx_block - c1 - x_hat * c2) * rstd_val
        
        # Store dx
        tl.store(dx_sample_ptr + col_offsets, dx_block, mask=mask)
    
    # Accumulate gradients for weight and bias
    if dweight_ptr is not None:
        for offset in range(0, n_elements_per_sample, BLOCK_SIZE_N):
            col_offsets = offset + tl.arange(0, BLOCK_SIZE_N)
            mask = col_offsets < n_elements_per_sample
            tl.atomic_add(dweight_ptr + col_offsets, dweight_acc, mask=mask)
    
    if dbias_ptr is not None:
        for offset in range(0, n_elements_per_sample, BLOCK_SIZE_N):
            col_offsets = offset + tl.arange(0, BLOCK_SIZE_N)
            mask = col_offsets < n_elements_per_sample
            tl.atomic_add(dbias_ptr + col_offsets, dbias_acc, mask=mask)

def triton_layer_norm_forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    """Triton wrapper for LayerNorm forward pass."""
    # Flatten the last three dimensions
    original_shape = x.shape
    n_samples = x.shape[0]
    n_elements_per_sample = x.numel() // n_samples
    
    # Reshape to 2D: (batch_size, -1)
    x_2d = x.view(n_samples, -1)
    
    # Output tensor
    output = torch.empty_like(x_2d)
    
    # Statistics for backward pass
    mean = torch.empty(n_samples, dtype=torch.float32, device=x.device)
    rstd = torch.empty(n_samples, dtype=torch.float32, device=x.device)
    
    # Flatten weight and bias
    weight_flat = weight.view(-1)
    bias_flat = bias.view(-1)
    
    # Kernel configuration
    BLOCK_SIZE_N = 1024  # Maximum threads per block
    
    # Grid configuration - one block per sample
    grid = lambda meta: (n_samples,)
    
    # Launch kernel
    layer_norm_forward_kernel[grid](
        x_2d,
        weight_flat,
        bias_flat,
        output,
        mean,
        rstd,
        n_elements_per_sample,
        x_2d.stride(0),
        weight_flat.stride(0),
        eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Reshape back to original shape
    return output.view(original_shape), mean, rstd

def triton_layer_norm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
):
    """Triton wrapper for LayerNorm backward pass."""
    # Flatten tensors
    n_samples = x.shape[0]
    n_elements_per_sample = x.numel() // n_samples
    
    x_2d = x.view(n_samples, -1)
    dy_2d = dy.view(n_samples, -1)
    weight_flat = weight.view(-1)
    
    # Gradient tensors
    dx = torch.empty_like(x_2d)
    dweight = torch.zeros_like(weight_flat)
    dbias = torch.zeros_like(weight_flat)
    
    # Kernel configuration
    BLOCK_SIZE_N = 1024
    
    # Grid configuration
    grid = lambda meta: (n_samples,)
    
    # Launch kernel
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
    )
    
    # Reshape gradients back to original shapes
    return dx.view(x.shape), dweight.view(weight.shape), dbias.view(weight.shape)

class ModelNew(nn.Module):
    """
    Optimized LayerNorm using Triton kernels.
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
