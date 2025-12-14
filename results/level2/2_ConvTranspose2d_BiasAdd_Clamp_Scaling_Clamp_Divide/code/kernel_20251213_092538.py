import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def post_process_kernel(
    x_ptr,
    bias_ptr,
    output_ptr,
    scaling_factor,
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    # 2D grid: (spatial_blocks, batch*channel)
    pid = tl.program_id(0)
    bc_id = tl.program_id(1)
    
    spatial_size = H * W
    batch_id = bc_id // C
    channel_id = bc_id % C
    base_offset = batch_id * C * spatial_size + channel_id * spatial_size
    
    # Process VEC_SIZE elements per thread, BLOCK_SIZE threads per block
    spatial_start = pid * BLOCK_SIZE * VEC_SIZE
    offsets = base_offset + spatial_start + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE + tl.arange(0, VEC_SIZE)[None, :]
    
    # Compute mask for boundary handling
    spatial_idx = tl.arange(0, BLOCK_SIZE)[:, None] * VEC_SIZE + tl.arange(0, VEC_SIZE)[None, :] + spatial_start
    mask = spatial_idx < spatial_size
    
    # Vectorized load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load bias and broadcast
    bias = tl.load(bias_ptr + channel_id)
    x = x + bias
    
    # Fused operations with improved numerical stability
    scale_inv = 1.0 / scaling_factor
    x = tl.minimum(tl.maximum(x, 0.0), 1.0)
    x_scaled = x * scaling_factor
    x = tl.minimum(tl.maximum(x_scaled, 0.0), 1.0) * scale_inv
    
    # Vectorized store
    tl.store(output_ptr + offsets, x, mask=mask)

def triton_post_process(
    x: torch.Tensor,
    bias: torch.Tensor,
    scaling_factor: float
) -> torch.Tensor:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert bias.is_contiguous(), "Bias tensor must be contiguous"
    
    B, C, H, W = x.shape
    spatial_size = H * W
    output = torch.empty_like(x)
    
    # Configurations with optimized num_warps and num_stages
    configs = [
        {'BLOCK_SIZE': 128, 'VEC_SIZE': 8, 'num_warps': 4, 'num_stages': 2},
        {'BLOCK_SIZE': 128, 'VEC_SIZE': 8, 'num_warps': 4, 'num_stages': 3},
        {'BLOCK_SIZE': 256, 'VEC_SIZE': 4, 'num_warps': 8, 'num_stages': 2},
        {'BLOCK_SIZE': 256, 'VEC_SIZE': 4, 'num_warps': 8, 'num_stages': 3},
        {'BLOCK_SIZE': 512, 'VEC_SIZE': 2, 'num_warps': 16, 'num_stages': 2},
        {'BLOCK_SIZE': 512, 'VEC_SIZE': 2, 'num_warps': 16, 'num_stages': 3},
        {'BLOCK_SIZE': 1024, 'VEC_SIZE': 1, 'num_warps': 32, 'num_stages': 1},
        {'BLOCK_SIZE': 1024, 'VEC_SIZE': 1, 'num_warps': 32, 'num_stages': 2},
    ]
    
    bias_reshaped = bias.view(-1)
    best_time = float('inf')
    best_config = None
    
    # Manual autotuning
    for config in configs:
        BLOCK_SIZE = config['BLOCK_SIZE']
        VEC_SIZE = config['VEC_SIZE']
        
        spatial_blocks = (spatial_size + BLOCK_SIZE * VEC_SIZE - 1) // (BLOCK_SIZE * VEC_SIZE)
        grid = (spatial_blocks, B * C)
        
        # Warmup
        post_process_kernel[grid](
            x, bias_reshaped, output,
            scaling_factor,
            B, C, H, W,
            BLOCK_SIZE=BLOCK_SIZE,
            VEC_SIZE=VEC_SIZE,
            num_warps=config['num_warps'],
            num_stages=config['num_stages']
        )
        
        # Benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(10):
            post_process_kernel[grid](
                x, bias_reshaped, output,
                scaling_factor,
                B, C, H, W,
                BLOCK_SIZE=BLOCK_SIZE,
                VEC_SIZE=VEC_SIZE,
                num_warps=config['num_warps'],
                num_stages=config['num_stages']
            )
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event) / 10.0
        
        if elapsed_time < best_time:
            best_time = elapsed_time
            best_config = config
    
    # Execute with best configuration
    BLOCK_SIZE = best_config['BLOCK_SIZE']
    VEC_SIZE = best_config['VEC_SIZE']
    spatial_blocks = (spatial_size + BLOCK_SIZE * VEC_SIZE - 1) // (BLOCK_SIZE * VEC_SIZE)
    grid = (spatial_blocks, B * C)
    
    post_process_kernel[grid](
        x, bias_reshaped, output,
        scaling_factor,
        B, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
        VEC_SIZE=VEC_SIZE,
        num_warps=best_config['num_warps'],
        num_stages=best_config['num_stages']
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        return triton_post_process(x, self.bias, self.scaling_factor)
