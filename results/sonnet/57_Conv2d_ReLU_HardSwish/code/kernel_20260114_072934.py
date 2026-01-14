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
    
    # Load input
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # Fused ReLU: max(x, 0)
    x_relu = tl.maximum(x, 0.0)
    
    # Fused HardSwish: x * clamp((x + 3) / 6, 0, 1)
    # Optimized: use fused multiply-add and minimize operations
    x_plus_3 = x_relu + 3.0
    # Multiply by 1/6 instead of divide by 6 (faster)
    x_scaled = x_plus_3 * 0.16666666666666666
    # Clamp to [0, 1]
    clamped = tl.minimum(tl.maximum(x_scaled, 0.0), 1.0)
    # Final multiply
    result = x_relu * clamped
    
    # Store result
    tl.store(out_ptr + offs, result, mask=mask)


def fused_relu_hardswish(x):
    N = x.numel()
    out = torch.empty_like(x)
    
    # Optimized block size for RTX 4090 (Ada architecture)
    # 256 provides good balance between occupancy and memory coalescing
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
