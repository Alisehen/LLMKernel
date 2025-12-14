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
):
    """
    Fused 3D convolution with pointwise operations (scaling, tanh, bias, sigmoid).
    """
    # Program ID mapping - Triton only supports 3D grid, so we flatten dimensions
    pid = tl.program_id(0)
    
    # Reconstruct indices from flattened pid
    # pid = b * (C_out * out_d * out_h * out_w) + c * (out_d * out_h * out_w) + 
    #       od * (out_h * out_w) + oh * out_w + ow
    pid_b = pid // (C_out * out_d * out_h * out_w)
    remaining = pid % (C_out * out_d * out_h * out_w)
    
    pid_c = remaining // (out_d * out_h * out_w)
    remaining = remaining % (out_d * out_h * out_w)
    
    pid_d = remaining // (out_h * out_w)
    remaining = remaining % (out_h * out_w)
    
    pid_h = remaining // out_w
    pid_w = remaining % out_w
    
    # Check bounds
    b_mask = pid_b < B
    c_mask = pid_c < C_out
    d_mask = pid_d < out_d
    h_mask = pid_h < out_h
    w_mask = pid_w < out_w
    
    # Only process if all indices are valid
    if not (b_mask & c_mask & d_mask & h_mask & w_mask):
        return
    
    # Calculate input starting positions
    in_d_start = pid_d * stride_d - padding_d
    in_h_start = pid_h * stride_h - padding_h
    in_w_start = pid_w * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over kernel depth dimension
    for kd in range(K_d):
        in_d = in_d_start + kd * dilation_d
        in_d_mask = (in_d >= 0) & (in_d < D)
        
        # Loop over kernel height dimension
        for kh in range(K_h):
            in_h = in_h_start + kh * dilation_h
            in_h_mask = (in_h >= 0) & (in_h < H)
            
            # Loop over kernel width dimension
            for kw in range(K_w):
                in_w = in_w_start + kw * dilation_w
                in_w_mask = (in_w >= 0) & (in_w < W)
                
                # Combined spatial mask for this kernel position
                spatial_valid = in_d_mask & in_h_mask & in_w_mask
                
                if spatial_valid:
                    # Loop over input channels in blocks
                    for c_block in range(0, C_in, BLOCK_SIZE_M):
                        c_offsets = c_block + tl.arange(0, BLOCK_SIZE_M)
                        c_mask = c_offsets < C_in
                        
                        # Calculate input indices - flatten 5D tensor
                        input_idx = (
                            pid_b * C_in * D * H * W +
                            c_offsets * D * H * W +
                            in_d * H * W +
                            in_h * W +
                            in_w
                        )
                        x = tl.load(x_ptr + input_idx, mask=c_mask, other=0.0)
                        
                        # Calculate weight indices
                        weight_idx = (
                            pid_c * C_in * K_d * K_h * K_w +
                            c_offsets * K_d * K_h * K_w +
                            kd * K_h * K_w +
                            kh * K_w +
                            kw
                        )
                        weight = tl.load(weight_ptr + weight_idx, mask=c_mask, other=0.0)
                        
                        # Accumulate
                        acc += x * weight
    
    # Sum across the channel dimension
    acc_sum = tl.sum(acc)
    
    # Load parameters for pointwise operations
    bias_val = tl.load(bias_ptr + pid_c)
    scaling = tl.load(scaling_factor_ptr + pid_c)
    bias2_val = tl.load(bias2_ptr + pid_c)
    
    # Apply pointwise operations
    result = acc_sum + bias_val
    result = result * scaling
    # tanh(x) = 2*sigmoid(2*x) - 1
    result = 2.0 * tl.sigmoid(2.0 * result) - 1.0
    result = result * bias2_val
    result = tl.sigmoid(result)
    
    # Calculate output index
    output_idx = (
        pid_b * C_out * out_d * out_h * out_w +
        pid_c * out_d * out_h * out_w +
        pid_d * out_h * out_w +
        pid_h * out_w +
        pid_w
    )
    
    # Store output
    tl.store(output_ptr + output_idx, result)

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
    out_d = (D + 2 * padding - dilation * (K_d - 1) - 1) // stride + 1
    out_h = (H + 2 * padding - dilation * (K_h - 1) - 1) // stride + 1
    out_w = (W + 2 * padding - dilation * (K_w - 1) - 1) // stride + 1
    
    output = torch.empty((B, C_out, out_d, out_h, out_w), 
                        device=x.device, dtype=x.dtype)
    
    # Grid configuration - flatten all output elements
    total_elements = B * C_out * out_d * out_h * out_w
    
    # Use optimal block sizes for memory access patterns
    BLOCK_SIZE_M = 32  # Input channel block size
    
    # Launch kernel with 1D grid
    grid = (total_elements,)
    
    conv_pointwise_kernel[grid](
        x, weight, bias, scaling_factor, bias2, output,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        groups,
        B, C_in, C_out, D, H, W, K_d, K_h, K_w,
        out_d, out_h, out_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=1,  # Not used but kept for compatibility
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
        scaling_factor = self.scaling_factor.view(-1, 1, 1, 1).squeeze()  # Shape: (out_channels,)
        bias = self.bias.view(-1, 1, 1, 1).squeeze()  # Shape: (out_channels,)
        
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
