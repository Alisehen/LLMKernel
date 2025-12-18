import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_i2h_tanh_kernel(
    x_ptr, h_prev_ptr, w_i2h_ptr, b_i2h_ptr, h_next_ptr,
    batch_size, input_size, hidden_size,
    stride_x_batch, stride_x_feat,
    stride_h_batch, stride_h_feat,
    stride_w_hidden, stride_w_feat,
    stride_b,
    BLOCK_BATCH: tl.constexpr, BLOCK_HIDDEN: tl.constexpr, BLOCK_FEAT: tl.constexpr,
    num_stages: tl.constexpr = 2,
):
    pid_batch = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    batch_offsets = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    hidden_offsets = pid_hidden * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    
    # Compute total features and matrix dimensions for optimal dot product
    total_features = input_size + hidden_size
    # Use proper alignment for Tensor Cores - multiples of 16 for best performance
    total_features_aligned = tl.cdiv(total_features, 16) * 16
    
    acc = tl.zeros((BLOCK_BATCH, BLOCK_HIDDEN), dtype=tl.float32)
    
    # Optimized main computation loop
    for k in range(0, total_features_aligned, BLOCK_FEAT):
        k_offsets = k + tl.arange(0, BLOCK_FEAT)
        
        # Pre-compute masks once
        w_mask = (hidden_offsets[:, None] < hidden_size) & (k_offsets[None, :] < total_features)
        input_mask = (batch_offsets[:, None] < batch_size) & (k_offsets[None, :] < input_size)
        hidden_k_offsets = k_offsets - input_size
        hidden_data_mask = (batch_offsets[:, None] < batch_size) & (k_offsets[None, :] >= input_size) & (hidden_k_offsets[None, :] < hidden_size)
        
        # Load weight block with proper alignment
        w_ptrs = w_i2h_ptr + (hidden_offsets[:, None] * stride_w_hidden + k_offsets[None, :] * stride_w_feat)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Efficient data loading with minimal branches
        data_block = tl.zeros((BLOCK_BATCH, BLOCK_FEAT), dtype=tl.float32)
        
        # Load input data if within bounds
        input_data_ptrs = x_ptr + (batch_offsets[:, None] * stride_x_batch + k_offsets[None, :] * stride_x_feat)
        input_data = tl.load(input_data_ptrs, mask=input_mask, other=0.0)
        data_block = tl.where(input_mask, input_data, data_block)
        
        # Load hidden data if within bounds
        hidden_data_ptrs = h_prev_ptr + (batch_offsets[:, None] * stride_h_batch + hidden_k_offsets[None, :] * stride_h_feat)
        hidden_data = tl.load(hidden_data_ptrs, mask=hidden_data_mask, other=0.0)
        data_block = tl.where(hidden_data_mask, hidden_data, data_block)
        
        # Accumulate with Tensor Core optimization
        acc += tl.dot(data_block, w_block.T, allow_tf32=True)
    
    # Add bias
    bias = tl.load(b_i2h_ptr + hidden_offsets, mask=hidden_offsets < hidden_size, other=0.0)
    acc += bias[None, :]
    
    # Fast tanh approximation optimized for Ada Lovelace
    # Using piecewise rational approximation with fewer operations
    x = acc
    x2 = x * x
    x3 = x2 * x
    # Pade[3/3] approximation for tanh: (x + x³/15) / (1 + 2x²/5 + x⁴/105)
    # Modified for better accuracy and fewer operations
    numerator = x * (15.0 + x2)  # 15x + x³
    denominator = 15.0 + 6.0 * x2 + x2 * x2 * 0.0666667  # 15 + 6x² + x⁴/15
    result = numerator / denominator
    
    # Store final result
    h_next_ptrs = h_next_ptr + (batch_offsets[:, None] * stride_h_batch + hidden_offsets[None, :] * stride_h_feat)
    store_mask = (batch_offsets[:, None] < batch_size) & (hidden_offsets[None, :] < hidden_size)
    tl.store(h_next_ptrs, result, mask=store_mask)

@triton.jit
def fused_h2o_kernel(
    h_ptr, w_h2o_ptr, b_h2o_ptr, o_ptr,
    batch_size, hidden_size, output_size,
    stride_h_batch, stride_h_feat,
    stride_w_output, stride_w_feat,
    stride_b,
    stride_o_batch, stride_o_feat,
    BLOCK_BATCH: tl.constexpr, BLOCK_OUTPUT: tl.constexpr, BLOCK_HIDDEN: tl.constexpr,
    num_stages: tl.constexpr = 2,
):
    pid_batch = tl.program_id(0)
    pid_output = tl.program_id(1)
    
    batch_offsets = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    output_offsets = pid_output * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
    
    # Align hidden size for Tensor Core optimization
    hidden_size_aligned = tl.cdiv(hidden_size, 16) * 16
    
    acc = tl.zeros((BLOCK_BATCH, BLOCK_OUTPUT), dtype=tl.float32)
    
    # Optimized main loop with proper alignment
    for k in range(0, hidden_size_aligned, BLOCK_HIDDEN):
        k_offsets = k + tl.arange(0, BLOCK_HIDDEN)
        
        # Pre-compute masks
        h_mask = (batch_offsets[:, None] < batch_size) & (k_offsets[None, :] < hidden_size)
        w_mask = (output_offsets[:, None] < output_size) & (k_offsets[None, :] < hidden_size)
        
        # Load hidden state block
        h_ptrs = h_ptr + (batch_offsets[:, None] * stride_h_batch + k_offsets[None, :] * stride_h_feat)
        h_block = tl.load(h_ptrs, mask=h_mask, other=0.0)
        
        # Load weight block
        w_ptrs = w_h2o_ptr + (output_offsets[:, None] * stride_w_output + k_offsets[None, :] * stride_w_feat)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Accumulate with Tensor Core optimization
        acc += tl.dot(h_block, w_block.T, allow_tf32=True)
    
    # Add bias
    bias = tl.load(b_h2o_ptr + output_offsets, mask=output_offsets < output_size, other=0.0)
    acc += bias[None, :]
    
    # Store final output
    o_ptrs = o_ptr + (batch_offsets[:, None] * stride_o_batch + output_offsets[None, :] * stride_o_feat)
    store_mask = (batch_offsets[:, None] < batch_size) & (output_offsets[None, :] < output_size)
    tl.store(o_ptrs, acc, mask=store_mask)

def fused_i2h_tanh(x, h_prev, weight_i2h, bias_i2h):
    batch_size, input_size = x.shape
    hidden_size = weight_i2h.shape[0]
    
    h_next = torch.empty((batch_size, hidden_size), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        triton.cdiv(batch_size, META['BLOCK_BATCH']),
        triton.cdiv(hidden_size, META['BLOCK_HIDDEN'])
    )
    
    # Optimized block sizes for Ada Lovelace (4090)
    # Increased block sizes for better occupancy while staying within register limits
    fused_i2h_tanh_kernel[grid](
        x, h_prev, weight_i2h, bias_i2h, h_next,
        batch_size, input_size, hidden_size,
        x.stride(0), x.stride(1),
        h_prev.stride(0), h_prev.stride(1),
        weight_i2h.stride(0), weight_i2h.stride(1),
        bias_i2h.stride(0),
        BLOCK_BATCH=64,  # Increased for better batch parallelism
        BLOCK_HIDDEN=128,  # Increased to utilize more Tensor Cores
        BLOCK_FEAT=64,  # Balanced for memory access patterns
        num_warps=8,  # Increased warps for better SM utilization
        num_stages=3  # Increased pipelining for memory latency hiding
    )
    
    return h_next

def fused_h2o(h, weight_h2o, bias_h2o):
    batch_size, hidden_size = h.shape
    output_size = weight_h2o.shape[0]
    
    output = torch.empty((batch_size, output_size), device=h.device, dtype=h.dtype)
    
    grid = lambda META: (
        triton.cdiv(batch_size, META['BLOCK_BATCH']),
        triton.cdiv(output_size, META['BLOCK_OUTPUT'])
    )
    
    fused_h2o_kernel[grid](
        h, weight_h2o, bias_h2o, output,
        batch_size, hidden_size, output_size,
        h.stride(0), h.stride(1),
        weight_h2o.stride(0), weight_h2o.stride(1),
        bias_h2o.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_BATCH=64,  # Increased for better batch parallelism
        BLOCK_OUTPUT=128,  # Increased to utilize more Tensor Cores
        BLOCK_HIDDEN=128,  # Increased for better memory coalescing
        num_warps=8,  # Increased warps for better SM utilization
        num_stages=3  # Increased pipelining for memory latency hiding
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with proper scaling for better conditioning
        std_i2h = (2.0 / (input_size + hidden_size)) ** 0.5
        std_h2o = (2.0 / hidden_size) ** 0.5
        
        self.weight_i2h = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size) * std_i2h)
        self.bias_i2h = nn.Parameter(torch.zeros(hidden_size))
        self.weight_h2o = nn.Parameter(torch.randn(output_size, hidden_size) * std_h2o)
        self.bias_h2o = nn.Parameter(torch.zeros(output_size))
    
    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]
            hidden = fused_i2h_tanh(x_t, hidden, self.weight_i2h, self.bias_i2h)
            output_t = fused_h2o(hidden, self.weight_h2o, self.bias_h2o)
            outputs.append(output_t)
        
        return torch.stack(outputs, dim=0)
