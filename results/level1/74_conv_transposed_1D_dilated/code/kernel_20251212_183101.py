import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional


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
    GROUP_C_IN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_l = tl.program_id(2)
    
    c_out_offset = pid_c * BLOCK_C_OUT + tl.arange(0, BLOCK_C_OUT)
    l_out_offset = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    
    c_out_mask = c_out_offset < C_out
    l_out_mask = l_out_offset < L_out
    
    acc = tl.zeros((BLOCK_C_OUT, BLOCK_L), dtype=tl.float32)
    
    # Process input channels in groups
    for c_in_block in range(0, C_in, GROUP_C_IN):
        c_in_base = c_in_block + tl.arange(0, GROUP_C_IN)
        c_in_mask = c_in_base < C_in
        
        # Load weight for all K elements at once
        for k in range(K):
            # Weight slice for this k: [GROUP_C_IN, BLOCK_C_OUT]
            w_ptrs = w_ptr + (
                c_in_base[:, None] * (C_out * K) +
                c_out_offset[None, :] * K +
                k
            )
            w_slice = tl.load(w_ptrs, 
                           mask=c_in_mask[:, None] & c_out_mask[None, :], 
                           other=0.0)
            
            # Compute input positions for this k
            l_in_pos = (l_out_offset - dilation * k + padding) // stride
            
            # Check bounds and stride condition
            valid_l_in = (l_in_pos >= 0) & (l_in_pos < L_in)
            valid_stride = ((l_out_offset - dilation * k + padding) % stride == 0)
            in_bounds = valid_l_in & valid_stride
            
            # Input pointers: [GROUP_C_IN, BLOCK_L]
            x_ptrs = x_ptr + (
                pid_b * (C_in * L_in) +
                c_in_base[:, None] * L_in +
                l_in_pos[None, :]
            )
            
            # Load input with broadcasting
            mask = c_in_mask[:, None] & in_bounds[None, :]
            x_tile = tl.load(x_ptrs, mask=mask, other=0.0)
            
            # Accumulate: w_slice.T @ x_tile -> [BLOCK_C_OUT, GROUP_C_IN] @ [GROUP_C_IN, BLOCK_L]
            acc += tl.dot(w_slice.T, x_tile, allow_tf32=True)
    
    if has_bias:
        b_ptrs = b_ptr + c_out_offset
        bias = tl.load(b_ptrs, mask=c_out_mask, other=0.0)
        acc += bias[:, None]
    
    # Store results
    out_ptrs = out_ptr + (
        pid_b * (C_out * L_out) +
        c_out_offset[:, None] * L_out +
        l_out_offset[None, :]
    )
    tl.store(out_ptrs, acc, mask=c_out_mask[:, None] & l_out_mask[None, :])


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
    
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    output = torch.empty((B, C_out, L_out), device=x.device, dtype=x.dtype)
    
    # Heuristics for block sizes
    BLOCK_C_IN = 16
    BLOCK_C_OUT = 64 if C_out >= 64 else 32 if C_out >= 32 else 16
    BLOCK_L = 128 if L_out >= 256 else 64 if L_out >= 64 else 32
    GROUP_C_IN = 16
    
    if C_in < 16:
        BLOCK_C_IN = 8
        GROUP_C_IN = 8
    
    grid = (B, triton.cdiv(C_out, BLOCK_C_OUT), triton.cdiv(L_out, BLOCK_L))
    
    conv_transpose1d_kernel[grid](
        x, weight,
        bias if bias is not None else torch.empty(0, device=x.device),
        output,
        B, C_in, C_out, L_in, L_out, K,
        stride, padding, dilation,
        has_bias=bias is not None,
        BLOCK_C_IN=BLOCK_C_IN,
        BLOCK_C_OUT=BLOCK_C_OUT,
        BLOCK_L=BLOCK_L,
        GROUP_C_IN=GROUP_C_IN,
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
