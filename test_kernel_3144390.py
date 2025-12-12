import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def pointwise_conv_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch, channels_in, channels_out, height, width,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic,
    stride_outb, stride_outc, stride_outh, stride_outw,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
):
    """
    Optimized pointwise 2D convolution kernel.
    
    Strategy:
    - Each program handles a tile of spatial dimensions (h,w) and a tile of output channels
    - Loop over input channels in groups for better memory locality
    - Use tensor cores via tl.dot when possible
    """
    # 3D grid: [batch * h_tiles, w_tiles, c_tiles]
    pid_bh = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_oc = tl.program_id(2)
    
    # Decompose batch and height indices
    grid_h = (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    batch_idx = pid_bh // grid_h
    h_tile_idx = pid_bh % grid_h
    
    # Offsets for this block
    h_start = h_tile_idx * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    oc_start = pid_oc * BLOCK_SIZE_C
    
    # Create masks for boundaries
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    oc_offsets = oc_start + tl.arange(0, BLOCK_SIZE_C)
    
    h_mask = h_offsets < height
    w_mask = w_offsets < width
    oc_mask = oc_offsets < channels_out
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)
    
    # Group input channels for better cache utilization
    num_groups = (channels_in + GROUP_SIZE_C - 1) // GROUP_SIZE_C
    
    for g in range(num_groups):
        ic_start = g * GROUP_SIZE_C
        
        # Load input tile [BLOCK_SIZE_H, BLOCK_SIZE_W, GROUP_SIZE_C]
        ic_offsets = ic_start + tl.arange(0, GROUP_SIZE_C)
        ic_mask = ic_offsets < channels_in
        
        x_ptrs = (
            x_ptr +
            batch_idx * stride_xb +
            h_offsets[:, None, None] * stride_xh +
            w_offsets[None, :, None] * stride_xw +
            ic_offsets[None, None, :] * stride_xc
        )
        
        mask_x = h_mask[:, None, None] & w_mask[None, :, None] & ic_mask[None, None, :]
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # Load weight tile [BLOCK_SIZE_C, GROUP_SIZE_C]
        w_ptrs = (
            w_ptr +
            oc_offsets[:, None] * stride_woc +
            ic_offsets[None, :] * stride_wic
        )
        mask_w = oc_mask[:, None] & ic_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        # Compute partial convolution using tensor operations
        # Reshape for efficient dot product
        x_flat = tl.reshape(x_tile, (BLOCK_SIZE_H * BLOCK_SIZE_W, GROUP_SIZE_C))
        w_flat = tl.reshape(w_tile, (BLOCK_SIZE_C, GROUP_SIZE_C))
        
        # Use tl.dot for tensor core acceleration
        partial = tl.dot(x_flat, w_flat.T)
        acc += tl.reshape(partial, (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    
    # Add bias if present
    if HAS_BIAS:
        b_ptrs = b_ptr + oc_offsets
        bias = tl.load(b_ptrs, mask=oc_mask, other=0.0)
        acc += bias[None, None, :]
    
    # Store output
    out_ptrs = (
        out_ptr +
        batch_idx * stride_outb +
        h_offsets[:, None, None] * stride_outh +
        w_offsets[None, :, None] * stride_outw +
        oc_offsets[None, None, :] * stride_outc
    )
    
    mask_out = h_mask[:, None, None] & w_mask[None, :, None] & oc_mask[None, None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


@triton.jit
def pointwise_conv_kernel_fp16(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    batch, channels_in, channels_out, height, width,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic,
    stride_outb, stride_outc, stride_outh, stride_outw,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    GROUP_SIZE_C: tl.constexpr,
):
    """
    FP16 optimized version with tensor core acceleration.
    """
    # Same 3D grid structure
    pid_bh = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_oc = tl.program_id(2)
    
    # Decompose batch and height
    grid_h = (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    batch_idx = pid_bh // grid_h
    h_tile_idx = pid_bh % grid_h
    
    # Offsets
    h_start = h_tile_idx * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W
    oc_start = pid_oc * BLOCK_SIZE_C
    
    # Masks
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
    oc_offsets = oc_start + tl.arange(0, BLOCK_SIZE_C)
    
    h_mask = h_offsets < height
    w_mask = w_offsets < width
    oc_mask = oc_offsets < channels_out
    
    # Initialize accumulator in fp32 for precision
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)
    
    # Process input channels in groups
    num_groups = (channels_in + GROUP_SIZE_C - 1) // GROUP_SIZE_C
    
    for g in range(num_groups):
        ic_start = g * GROUP_SIZE_C
        
        # Load input tile in fp16
        ic_offsets = ic_start + tl.arange(0, GROUP_SIZE_C)
        ic_mask = ic_offsets < channels_in
        
        x_ptrs = (
            x_ptr +
            batch_idx * stride_xb +
            h_offsets[:, None, None] * stride_xh +
            w_offsets[None, :, None] * stride_xw +
            ic_offsets[None, None, :] * stride_xc
        )
        
        mask_x = h_mask[:, None, None] & w_mask[None, :, None] & ic_mask[None, None, :]
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # Load weight tile in fp16
        w_ptrs = (
            w_ptr +
            oc_offsets[:, None] * stride_woc +
            ic_offsets[None, :] * stride_wic
        )
        mask_w = oc_mask[:, None] & ic_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        # Convert to fp16 for tensor core operation
        x_fp16 = x_tile.to(tl.float16)
        w_fp16 = w_tile.to(tl.float16)
        
        # Reshape for tensor core dot product
        x_flat = tl.reshape(x_fp16, (BLOCK_SIZE_H * BLOCK_SIZE_W, GROUP_SIZE_C))
        w_flat = tl.reshape(w_fp16, (BLOCK_SIZE_C, GROUP_SIZE_C))
        
        # Tensor core accelerated dot product (fp16->fp32 accumulation)
        partial = tl.dot(x_flat, w_flat.T, out_dtype=tl.float32)
        acc += tl.reshape(partial, (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C))
    
    # Add bias
    if HAS_BIAS:
        b_ptrs = b_ptr + oc_offsets
        bias = tl.load(b_ptrs, mask=oc_mask, other=0.0)
        acc += bias[None, None, :]
    
    # Store output
    out_ptrs = (
        out_ptr +
        batch_idx * stride_outb +
        h_offsets[:, None, None] * stride_outh +
        w_offsets[None, :, None] * stride_outw +
        oc_offsets[None, None, :] * stride_outc
    )
    
    mask_out = h_mask[:, None, None] & w_mask[None, :, None] & oc_mask[None, None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


def triton_pointwise_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    Pointwise 2D convolution using Triton kernels.
    """
    batch, channels_in, height, width = x.shape
    channels_out = weight.shape[0]
    
    # Allocate output
    out = torch.empty(
        (batch, channels_out, height, width),
        device=x.device,
        dtype=x.dtype
    )
    
    # Get strides
    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_outb, stride_outc, stride_outh, stride_outw = out.stride()
    stride_woc, stride_wic = weight.stride(0), weight.stride(1)
    
    # Choose kernel based on dtype
    if x.dtype in (torch.float16, torch.bfloat16):
        kernel = pointwise_conv_kernel_fp16
        # Optimal tile sizes for tensor cores on Ampere
        BLOCK_SIZE_H = 16
        BLOCK_SIZE_W = 16
        BLOCK_SIZE_C = 64
        GROUP_SIZE_C = 32
    else:
        kernel = pointwise_conv_kernel
        # Optimal tile sizes for fp32
        BLOCK_SIZE_H = 16
        BLOCK_SIZE_W = 16
        BLOCK_SIZE_C = 32
        GROUP_SIZE_C = 16
    
    # Grid dimensions (use standard integer arithmetic)
    grid_h = (height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (channels_out + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # 3D grid: [batch * h_tiles, w_tiles, c_tiles]
    grid = (batch * grid_h, grid_w, grid_c)
    
    # Handle bias pointer
    if bias is not None:
        bias_ptr = bias
        has_bias = True
    else:
        # Create dummy bias tensor with correct dtype and device
        bias_ptr = torch.zeros(1, device=x.device, dtype=x.dtype)
        has_bias = False
    
    # Launch kernel
    kernel[grid](
        x,
        weight,
        bias_ptr,
        out,
        batch, channels_in, channels_out, height, width,
        stride_xb, stride_xc, stride_xh, stride_xw,
        stride_woc, stride_wic,
        stride_outb, stride_outc, stride_outh, stride_outw,
        HAS_BIAS=has_bias,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        GROUP_SIZE_C=GROUP_SIZE_C,
    )
    
    return out


class ModelNew(nn.Module):
    """
    Performs a pointwise 2D convolution operation using Triton kernels.
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, 1, 1)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Kaiming initialization
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze the spatial dimensions from weight
        weight_2d = self.weight.squeeze(-1).squeeze(-1)
        
        return triton_pointwise_conv2d(
            x,
            weight_2d,
            self.bias
        )
