import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_min_clamp_dropout_kernel(
    x_ptr, output_ptr, 
    n_elements,
    min_value, max_value,
    dropout_p, seed,
    training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Min operation: x = min(x, min_value)
    x = tl.minimum(x, min_value)
    
    # Clamp operation: x = clamp(x, min_value, max_value)
    x = tl.maximum(x, min_value)
    x = tl.minimum(x, max_value)
    
    # Dropout (only during training)
    if training:
        random = tl.rand(seed, offsets)
        dropout_mask = random > dropout_p
        scale = 1.0 / (1.0 - dropout_p)
        x = tl.where(dropout_mask, x * scale, 0.0)
    
    tl.store(output_ptr + offsets, x, mask=mask)


def fused_min_clamp_dropout(x, min_value, max_value, dropout_p, training):
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    seed = torch.randint(0, 2**31 - 1, (1,), device=x.device).item() if training else 0
    
    fused_min_clamp_dropout_kernel[grid](
        x, output,
        n_elements,
        min_value, max_value,
        dropout_p, seed,
        training,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


class ModelNew(nn.Module):
    """
    Optimized model with fused min, clamp, and dropout operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.min_value = min_value
        self.max_value = max_value
        self.dropout_p = dropout_p

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = fused_min_clamp_dropout(x, self.min_value, self.max_value, self.dropout_p, self.training)
        return x
