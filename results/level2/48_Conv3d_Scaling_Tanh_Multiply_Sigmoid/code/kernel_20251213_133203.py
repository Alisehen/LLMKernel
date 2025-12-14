import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def conv_pointwise_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    scaling_factor_ptr,
    bias2_ptr,
    output_ptr,
    stride_d, stride_h, stride_w,
    padding_d, padding_h, padding_w,
    dilation_d, dilation_h, dilation_w,
    groups,
    B, C_in, C_out, D, H, W, K_d, K_h, K_w,
    out_d, out_h, out_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused 3D convolution with pointwise operations (scaling, tanh, bias, sigmoid).
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_od = tl.program_id(2)
    pid_oh = tl.program_id(3)
    
    # Calculate starting output positions
    start_oh = pid_oh * BLOCK_SIZE_N
    start_ow = tl.arange(0, BLOCK_SIZE_K)
    
    # Create masks for spatial dimensions
    mask_oh = start_oh + tl.arange(0, BLOCK_SIZE_N)[:, None] < out_h
    mask_ow = start_ow[None, :] < out_w
    spatial_mask = mask_oh & mask_ow
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Calculate output depth position
    od = pid_od
    oh = start_oh + tl.arange(0, BLOCK_SIZE_N)[:, None]
    ow = start_ow[None, :]
    
    # Compute input starting positions for this output position
    in_d_start = od * stride_d - padding_d
    in_h_start = oh * stride_h - padding_h
    in_w_start = ow * stride_w - padding_w
    
    # Loop over kernel depth dimension
    for kd in range(K_d):
        # Loop over kernel height dimension  
        for kh in range(K_h):
            # Loop over kernel width dimension
            for kw in range(K_w):
                # Loop over input channels in blocks
                for c_block in range(0, C_in, BLOCK_SIZE_M):
                    c_offsets = c_block + tl.arange(0, BLOCK_SIZE_M)
                    c_mask = c_offsets < C_in
                    
                    # Compute input positions
                    in_d = in_d_start + kd * dilation_d
                    in_h = in_h_start + kh * dilation_h
                    in_w = in_w_start + kw * dilation_w
                    
                    # Create input mask
                    in_mask = (
                        (in_d >= 0) & (in_d < D) &
                        (in_h >= 0) & (in_h < H) &
                        (in_w >= 0) & (in_w < W)
                    ) & spatial_mask & c_mask[:, None, None]
                    
                    # Load input
                    in_idx = (
                        pid_b * C_in * D * H * W +
                        c_offsets[:, None, None] * D * H * W +
                        in_d * H * W +
                        in_h * W +
                        in_w
                    )
                    x = tl.load(
                        x_ptr + in_idx,
                        mask=in_mask,
                        other=0.0
                    )
                    
                    # Load weight
                    weight_idx = (
                        pid_c * (C_in // groups) * K_d * K_h * K_w +
                        c_offsets[:, None, None] * K_d * K_h * K_w +
                        kd * K_h * K_w +
                        kh * K_w +
                        kw
                    )
                    weight = tl.load(
                        weight_ptr + weight_idx,
                        mask=c_mask[:, None, None],
                        other=0.0
                    )
                    
                    # Accumulate
                    acc += tl.sum(x * weight, axis=0)
    
    # Load parameters for pointwise operations
    bias_val = tl.load(bias_ptr + pid_c)
    scaling = tl.load(scaling_factor_ptr + pid_c)
    bias2_val = tl.load(bias2_ptr + pid_c)
    
    # Apply pointwise operations (using triton's built-in functions)
    acc = acc + bias_val
    acc = acc * scaling
    # tanh(x) = 2*sigmoid(2*x) - 1 (triton doesn't have direct tanh)
    acc = 2.0 * tl.sigmoid(2.0 * acc) - 1.0
    acc = acc * bias2_val
    acc = tl.sigmoid(acc)
    
    # Store output
    out_idx = (
        pid_b * C_out * out_d * out_h * out_w +
        pid_c * out_d * out_h * out_w +
        od * out_h * out_w +
        (start_oh + tl.arange(0, BLOCK_SIZE_N)[:, None]) * out_w +
        start_ow[None, :]
    )
    
    tl.store(
        output_ptr + out_idx,
        acc,
        mask=spatial_mask
    )

def triton_conv_pointwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: torch.Tensor,
    bias2: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    """
    Wrapper function for the fused convolution-pointwise Triton kernel.
    """
    B, C_in, D, H, W = x.shape
    C_out, C_in_groups, K_d, K_h, K_w = weight.shape
    
    # Calculate output dimensions
    out_d = (D + 2*padding - dilation*(K_d-1) - 1) // stride + 1
    out_h = (H + 2*padding - dilation*(K_h-1) - 1) // stride + 1
    out_w = (W + 2*padding - dilation*(K_w-1) - 1) // stride + 1
    
    output = torch.empty((B, C_out, out_d, out_h, out_w), 
                        device=x.device, dtype=x.dtype)
    
    # Grid configuration - 4D grid for batch, channels, depth, and height blocks
    grid_oh = triton.cdiv(out_h, 16)
    grid = (B, C_out, out_d, grid_oh)
    
    # Launch kernel with optimal block sizes
    conv_pointwise_kernel[grid](
        x, weight, bias, scaling_factor, bias2, output,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        groups,
        B, C_in, C_out, D, H, W, K_d, K_h, K_w,
        out_d, out_h, out_w,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16,
    )
    
    return output

class ModelNew(nn.Module):
    """
    Optimized model using fused Triton kernel for 3D convolution and pointwise operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Store convolution parameters
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = kernel_size // 2
        self.dilation = 1
        self.groups = 1

    def forward(self, x):
        # Ensure tensors are contiguous
        x = x.contiguous()
        
        # Reshape parameters if needed
        scaling_factor = self.scaling_factor.view(-1, 1, 1, 1)
        bias = self.bias.view(-1, 1, 1, 1)
        
        return triton_conv_pointwise(
            x, 
            self.conv_weight.contiguous(), 
            self.conv_bias, 
            scaling_factor, 
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
