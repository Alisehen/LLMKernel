import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'VECTORIZE': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'VECTORIZE': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024, 'VECTORIZE': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048, 'VECTORIZE': 2}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def mish_tanh_kernel_3d_optimized(
    input_ptr,
    output_ptr,
    n_elements,
    B: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VECTORIZE: tl.constexpr,
):
    """
    Optimized kernel with 3D grid for better parallelism.
    Grid layout: (B * C, triton.cdiv(D * H * W, BLOCK_SIZE))
    """
    # Reconstruct 3D grid indices
    pid_bc = tl.program_id(0)  # batch*channel dimension
    pid_spatial = tl.program_id(1)  # spatial dimension
    
    batch_idx = pid_bc // C
    channel_idx = pid_bc % C
    
    # Calculate spatial base offset
    spatial_size = D * H * W
    spatial_start = pid_spatial * BLOCK_SIZE
    
    # Vectorized loading for better memory throughput
    if VECTORIZE == 2:
        offsets_outer = tl.arange(0, BLOCK_SIZE // 2) * 2
        offsets_inner = tl.arange(0, 2)
        offsets = spatial_start + offsets_outer[:, None] + offsets_inner[None, :]
        offsets = tl.reshape(offsets, (-1,))
    else:
        offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    
    # Apply mask for spatial dimension
    mask = offsets < spatial_size
    
    # Calculate base pointer offset for this batch and channel
    base_offset = ((batch_idx * C + channel_idx) * spatial_size)
    input_offsets = base_offset + offsets
    output_offsets = base_offset + offsets
    
    # Load input only where mask is valid
    x = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Optimized Mish+Tanh with fast math approximations
    # Use exp2 for faster exponential computation
    # Clamp to avoid overflow in exponential
    x_clamped = tl.where(x > 20.0, 20.0, x)
    x_clamped = tl.where(x_clamped < -20.0, -20.0, x_clamped)
    
    # Mish activation: x * tanh(softplus(x))
    # Softplus: log(1 + exp(x))
    # Use fast approximation for large values
    LOG2E = 1.442695  # 1/ln(2)
    LN2 = 0.693147    # ln(2)
    
    # Compute softplus(x) = log(1 + exp(x))
    exp_x = tl.exp2(x_clamped * LOG2E)
    exp_neg_x = tl.exp2(-x_clamped * LOG2E)
    
    # For numerical stability, compute log(1+exp(x)) differently for positive/negative x
    softplus_x = tl.where(
        x_clamped > 0,
        x_clamped + tl.log2(1.0 + exp_neg_x) * LN2,
        tl.log2(1.0 + exp_x) * LN2
    )
    
    # tanh(softplus_x) using optimized formula: (exp(2x) - 1)/(exp(2x) + 1)
    exp_2sp = tl.exp2(2.0 * softplus_x * LOG2E)
    tanh_sp = (exp_2sp - 1.0) / (exp_2sp + 1.0)
    mish_x = x * tanh_sp
    
    # Tanh activation on mish output
    mish_clamped = tl.where(mish_x > 20.0, 20.0, mish_x)
    mish_clamped = tl.where(mish_clamped < -20.0, -20.0, mish_clamped)
    
    exp_2mish = tl.exp2(2.0 * mish_clamped * LOG2E)
    result = (exp_2mish - 1.0) / (exp_2mish + 1.0)
    
    # Store result only where mask is valid
    tl.store(output_ptr + output_offsets, result, mask=mask)


def triton_mish_tanh_3d_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized wrapper with 3D grid layout for better SM utilization.
    """
    output = torch.empty_like(x)
    
    # Get tensor dimensions
    B, C, D, H, W = x.shape
    n_elements = x.numel()
    
    # Ensure tensor is contiguous
    x = x.contiguous()
    
    # Use 2D grid: (B*C, cdiv(D*H*W, BLOCK_SIZE))
    spatial_size = D * H * W
    grid = lambda meta: (
        B * C,
        triton.cdiv(spatial_size, meta['BLOCK_SIZE']),
    )
    
    mish_tanh_kernel_3d_optimized[grid](
        x, output, n_elements,
        B=B, C=C, D=D, H=H, W=W
    )
    return output


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    Uses optimized Triton kernel with 3D grid layout for better parallelism.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        x = self.conv(x)
        x = triton_mish_tanh_3d_optimized(x)
        return x
