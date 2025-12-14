import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional


@triton.autotune(
    configs=[
        # Optimized for tensor cores with better warp occupancy
        triton.Config({'BLOCK_C_IN': 16, 'BLOCK_C_OUT': 128, 'BLOCK_L': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_C_IN': 16, 'BLOCK_C_OUT': 128, 'BLOCK_L': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C_IN': 16, 'BLOCK_C_OUT': 256, 'BLOCK_L': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C_IN': 32, 'BLOCK_C_OUT': 128, 'BLOCK_L': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_C_IN': 32, 'BLOCK_C_OUT': 128, 'BLOCK_L': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C_IN': 32, 'BLOCK_C_OUT': 256, 'BLOCK_L': 64}, num_stages=3, num_warps=8),
    ],
    key=['C_in', 'C_out', 'L_out'],
)
@triton.jit
def conv_transpose1d_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B,
    C_in,
    C_out,
    L_in,
    L_out,
    K,
    stride,
    padding,
    dilation,
    has_bias: tl.constexpr,
    BLOCK_C_IN: tl.constexpr,
    BLOCK_C_OUT: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    pid_l_out = tl.program_id(2)
    
    # Offsets for the output block
    c_out_offsets = pid_c_out * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)
    l_out_offsets = pid_l_out * BLOCK_L + tl.arange(0, BLOCK_L)
    
    c_out_mask = c_out_offsets < C_out
    l_out_mask = l_out_offsets < L_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_C_OUT, BLOCK_L), dtype=tl.float32)
    
    # Iterate over input channels in blocks
    for c_in_idx in range(0, C_in, BLOCK_C_IN):
        c_in_offsets = c_in_idx + tl.arange(0, BLOCK_C_IN)
        c_in_mask = c_in_offsets < C_in
        
        # Iterate over kernel positions
        for k in range(K):
            # Load weight block for this kernel position
            w_offsets = (
                (c_in_offsets[:, None] * C_out + c_out_offsets[None, :]) * K + k
            )
            w = tl.load(
                w_ptr + w_offsets,
                mask=c_in_mask[:, None] & c_out_mask[None, :],
                other=0.0
            )
            
            # Calculate input position for this kernel position
            k_effective = k * dilation
            l_in_pos = (l_out_offsets - k_effective + padding)
            
            # Check if input position is valid
            if stride > 1:
                # For stride > 1, we need to check if l_in_pos is divisible by stride
                l_in_idx = l_in_pos // stride
                valid = (l_in_pos >= 0) & (l_in_idx < L_in) & (l_in_pos % stride == 0)
            else:
                # For stride = 1, just check bounds
                l_in_idx = l_in_pos
                valid = (l_in_pos >= 0) & (l_in_pos < L_in)
            
            # Load input block
            x_offsets = (
                pid_b * C_in * L_in +
                c_in_offsets[:, None] * L_in +
                tl.where(valid, l_in_idx, 0)[None, :]
            )
            x = tl.load(
                x_ptr + x_offsets,
                mask=c_in_mask[:, None] & valid[None, :],
                other=0.0
            )
            
            # Accumulate: w^T * x
            acc += tl.dot(w, x, allow_tf32=True)
    
    # Add bias if present
    if has_bias:
        b = tl.load(
            b_ptr + c_out_offsets,
            mask=c_out_mask,
            other=0.0
        )
        acc += b[:, None]
    
    # Store results
    out_offsets = (
        pid_b * C_out * L_out +
        c_out_offsets[:, None] * L_out +
        l_out_offsets[None, :]
    )
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=c_out_mask[:, None] & l_out_mask[None, :]
    )


def triton_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    B, C_in, L_in = x.shape
    C_out = weight.shape[1]
    K = weight.shape[2]
    
    # Calculate output length
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    
    # Prepare output tensor
    output = torch.empty((B, C_out, L_out), device=x.device, dtype=x.dtype)
    
    # Calculate grid
    grid = lambda META: (
        B,
        triton.cdiv(C_out, META['BLOCK_C_OUT']),
        triton.cdiv(L_out, META['BLOCK_L']),
    )
    
    # Launch kernel
    conv_transpose1d_kernel[grid](
        x, weight,
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype),
        output,
        B, C_in, C_out, L_in, L_out, K,
        stride, padding, dilation,
        has_bias=bias is not None,
    )
    
    return output


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Weight shape for conv_transpose1d: (in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            fan_in = in_channels * kernel_size
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose1d(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation
        )
