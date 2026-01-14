import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_relu_hardswish_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Load input with vectorization hint
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # Fused ReLU: max(x, 0)
    x_relu = tl.maximum(x, 0.0)
    
    # Fused HardSwish: x * clamp((x + 3) / 6, 0, 1)
    # Recompute x_relu in expression to reduce register pressure
    # Use multiply by constant instead of division
    x_plus_3 = x_relu + 3.0
    x_scaled = x_plus_3 * 0.16666666666666666
    
    # Clamp using min/max - these are cheap ops
    clamped = tl.minimum(tl.maximum(x_scaled, 0.0), 1.0)
    
    # Final result - recompute x_relu to save registers
    result = tl.maximum(x, 0.0) * clamped
    
    # Store result with vectorization
    tl.store(out_ptr + offs, result, mask=mask)


@triton.jit
def fused_relu_hardswish_kernel_large(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # Compute ReLU once
    x_relu = tl.maximum(x, 0.0)
    
    # HardSwish computation optimized for larger blocks
    # (x + 3) / 6 = x * (1/6) + 0.5
    x_scaled = x_relu * 0.16666666666666666 + 0.5
    
    # Clamp to [0, 1]
    clamped = tl.minimum(tl.maximum(x_scaled, 0.0), 1.0)
    
    # Final multiply
    result = x_relu * clamped
    
    # Store result
    tl.store(out_ptr + offs, result, mask=mask)


def fused_relu_hardswish(x):
    N = x.numel()
    out = torch.empty_like(x)
    
    # Autotune block size based on problem size
    # Larger blocks for better cache utilization and reduced launch overhead
    if N >= 1024 * 1024:
        # Large tensors: use 1024 for maximum memory coalescing
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        fused_relu_hardswish_kernel_large[grid](
            x, out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    elif N >= 256 * 1024:
        # Medium tensors: use 512
        BLOCK_SIZE = 512
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        fused_relu_hardswish_kernel[grid](
            x, out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        # Small tensors: use 256
        BLOCK_SIZE = 256
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        fused_relu_hardswish_kernel[grid](
            x, out,
            N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = fused_relu_hardswish(x)
        return x
