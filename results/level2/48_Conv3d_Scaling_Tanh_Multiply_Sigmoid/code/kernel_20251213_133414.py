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
    grid_oh: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused 3D convolution with pointwise operations (scaling, tanh, bias, sigmoid).
    """
    # 3D grid mapping: (batch * channels, depth * height_blocks, width_blocks)
    pid_bc = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_w = tl.program_id(2)
    
    # Decompose batch and channel from combined dimension
    pid_b = pid_bc // C_out
    pid_c = pid_bc % C_out
    
    # Decompose depth and height block from combined dimension
    pid_d = pid_dh // grid_oh
    pid_hb = pid_dh % grid_oh  # height block index
    
    # Calculate spatial positions
    start_h = pid_hb * BLOCK_SIZE_N
    start_w = pid_w * BLOCK_SIZE_K
    
    # Create masks for spatial dimensions
    oh = start_h + tl.arange(0, BLOCK_SIZE_N)
    ow = start_w + tl.arange(0, BLOCK_SIZE_K)
    mask_h = oh < out_h
    mask_w = ow < out_w
    spatial_mask = mask_h[:, None] & mask_w[None, :]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Compute input starting positions for this output position
    in_d_start = pid_d * stride_d - padding_d
    in_h_start = oh[:, None] * stride_h - padding_h
    in_w_start = ow[None, :] * stride_w - padding_w
    
    # Loop over kernel dimensions
    for kd in range(K_d):
        in_d = in_d_start + kd * dilation_d
        in_d_mask = (in_d >= 0) & (in_d < D)
        
        for kh in range(K_h):
            in_h = in_h_start + kh * dilation_h
            in_h_mask = (in_h >= 0) & (in_h < H)
            
            for kw in range(K_w):
                in_w = in_w_start + kw * dilation_w
                in_w_mask = (in_w >= 0) & (in_w < W)
                
                # Combined spatial mask
                spatial_valid = in_d_mask & in_h_mask & in_w_mask
                
                # Loop over input channels in blocks
                for c_block in range(0, C_in, BLOCK_SIZE_M):
                    c_offsets = c_block + tl.arange(0, BLOCK_SIZE_M)
                    c_mask = c_offsets < C_in
                    
                    # Load input with broadcasting mask
                    input_mask = c_mask[:, None, None] & spatial_valid & spatial_mask
                    
                    # Calculate input indices
                    input_idx = (
                        pid_b * C_in * D * H * W +
                        c_offsets[:, None, None] * D * H * W +
                        in_d * H * W +
                        in_h * W +
                        in_w
                    )
                    x = tl.load(x_ptr + input_idx, mask=input_mask, other=0.0)
                    
                    # Load weight
                    weight_idx = (
                        pid_c * C_in * K_d * K_h * K_w +
                        c_offsets[:, None, None] * K_d * K_h * K_w +
                        kd * K_h * K_w +
                        kh * K_w +
                        kw
                    )
                    weight = tl.load(weight_ptr + weight_idx, 
                                   mask=c_mask[:, None, None], other=0.0)
                    
                    # Accumulate
                    acc += tl.sum(x * weight, axis=0)
    
    # Load parameters for pointwise operations
    bias_val = tl.load(bias_ptr + pid_c)
    scaling = tl.load(scaling_factor_ptr + pid_c)
    bias2_val = tl.load(bias2_ptr + pid_c)
    
    # Apply pointwise operations
    acc = acc + bias_val
    acc = acc * scaling
    # tanh(x) = 2*sigmoid(2*x) - 1
    acc = 2.0 * tl.sigmoid(2.0 * acc) - 1.0
    acc = acc * bias2_val
    acc = tl.sigmoid(acc)
    
    # Store output
    output_idx = (
        pid_b * C_out * out_d * out_h * out_w +
        pid_c * out_d * out_h * out_w +
        pid_d * out_h * out_w +
        oh[:, None] * out_w +
        ow[None, :]
    )
    
    tl.store(output_ptr + output_idx, acc, mask=spatial_mask)

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
    
    # Grid configuration - 3D grid as required by Triton
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    grid_oh = (out_h + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_ow = (out_w + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    grid = (B * C_out, out_d * grid_oh, grid_ow)
    
    # Launch kernel
    conv_pointwise_kernel[grid](
        x, weight, bias, scaling_factor, bias2, output,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,
        groups,
        B, C_in, C_out, D, H, W, K_d, K_h, K_w,
        out_d, out_h, out_w,
        grid_oh,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
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
