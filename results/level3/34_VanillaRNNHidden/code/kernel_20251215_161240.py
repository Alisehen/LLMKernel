import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_i2h_tanh_kernel_optimized(
    x_ptr, h_prev_ptr, w_i2h_ptr, b_i2h_ptr, h_next_ptr,
    batch_size, input_size, hidden_size,
    stride_x_batch, stride_x_feat,
    stride_h_batch, stride_h_feat,
    stride_w_hidden, stride_w_feat,
    stride_b,
    BLOCK_BATCH: tl.constexpr, BLOCK_HIDDEN: tl.constexpr, BLOCK_FEAT: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    batch_offsets = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    hidden_offsets = pid_hidden * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    
    acc = tl.zeros((BLOCK_BATCH, BLOCK_HIDDEN), dtype=tl.float32)
    
    total_features = input_size + hidden_size
    
    for k in range(0, total_features, BLOCK_FEAT):
        k_offsets = k + tl.arange(0, BLOCK_FEAT)
        
        # Load weight block once
        w_mask = (hidden_offsets[:, None] < hidden_size) & (k_offsets[None, :] < total_features)
        w_ptrs = w_i2h_ptr + hidden_offsets[:, None] * stride_w_hidden + k_offsets[None, :] * stride_w_feat
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Efficient data loading with minimal conditionals
        data_block = tl.zeros((BLOCK_BATCH, BLOCK_FEAT), dtype=tl.float32)
        
        # Load input features
        if k < input_size:
            input_mask = (batch_offsets[:, None] < batch_size) & (k_offsets[None, :] < input_size)
            input_ptrs = x_ptr + batch_offsets[:, None] * stride_x_batch + k_offsets[None, :] * stride_x_feat
            input_data = tl.load(input_ptrs, mask=input_mask, other=0.0)
            data_block = tl.where(k_offsets[None, :] < input_size, input_data, data_block)
        
        # Load hidden features (only for k >= input_size)
        if k + BLOCK_FEAT > input_size:
            hidden_k_offsets = k_offsets - input_size
            hidden_mask = (batch_offsets[:, None] < batch_size) & (k_offsets[None, :] >= input_size) & (hidden_k_offsets[None, :] < hidden_size)
            hidden_ptrs = h_prev_ptr + batch_offsets[:, None] * stride_h_batch + hidden_k_offsets[None, :] * stride_h_feat
            hidden_data = tl.load(hidden_ptrs, mask=hidden_mask, other=0.0)
            data_block = tl.where((k_offsets[None, :] >= input_size) & (hidden_k_offsets[None, :] < hidden_size), hidden_data, data_block)
        
        # Accumulate with fused dot product
        acc += tl.dot(data_block, w_block.T, allow_tf32=USE_TF32)
    
    # Add bias and apply tanh
    bias = tl.load(b_i2h_ptr + hidden_offsets, mask=hidden_offsets < hidden_size, other=0.0)
    acc += bias[None, :]
    
    # Fast tanh approximation for moderate values
    x = tl.minimum(tl.maximum(acc, -9.0), 9.0)  # More aggressive clipping
    x2 = x * x
    x3 = x2 * x
    tanh_approx = x - (x3 * (1.0/3.0)) + (x3 * x2 * (2.0/15.0))
    result = tl.where(tl.abs(x) > 3.0, tl.math.tanh(x), tanh_approx)
    
    # Store
    h_next_ptrs = h_next_ptr + batch_offsets[:, None] * stride_h_batch + hidden_offsets[None, :] * stride_h_feat
    store_mask = (batch_offsets[:, None] < batch_size) & (hidden_offsets[None, :] < hidden_size)
    tl.store(h_next_ptrs, result, mask=store_mask)

@triton.jit
def fused_h2o_kernel_optimized(
    h_ptr, w_h2o_ptr, b_h2o_ptr, o_ptr,
    batch_size, hidden_size, output_size,
    stride_h_batch, stride_h_feat,
    stride_w_output, stride_w_feat,
    stride_b,
    stride_o_batch, stride_o_feat,
    BLOCK_BATCH: tl.constexpr, BLOCK_OUTPUT: tl.constexpr, BLOCK_HIDDEN: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_output = tl.program_id(1)
    
    batch_offsets = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    output_offsets = pid_output * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
    
    acc = tl.zeros((BLOCK_BATCH, BLOCK_OUTPUT), dtype=tl.float32)
    
    # Pre-compute bias to reduce register pressure
    bias = tl.load(b_h2o_ptr + output_offsets, mask=output_offsets < output_size, other=0.0)
    
    for k in range(0, hidden_size, BLOCK_HIDDEN):
        k_offsets = k + tl.arange(0, BLOCK_HIDDEN)
        
        # Load h block
        h_mask = (batch_offsets[:, None] < batch_size) & (k_offsets[None, :] < hidden_size)
        h_ptrs = h_ptr + batch_offsets[:, None] * stride_h_batch + k_offsets[None, :] * stride_h_feat
        h_block = tl.load(h_ptrs, mask=h_mask, other=0.0)
        
        # Load w block
        w_mask = (output_offsets[:, None] < output_size) & (k_offsets[None, :] < hidden_size)
        w_ptrs = w_h2o_ptr + output_offsets[:, None] * stride_w_output + k_offsets[None, :] * stride_w_feat
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Fused accumulation
        acc += tl.dot(h_block, w_block.T, allow_tf32=USE_TF32)
    
    # Add bias
    acc += bias[None, :]
    
    # Store
    o_ptrs = o_ptr + batch_offsets[:, None] * stride_o_batch + output_offsets[None, :] * stride_o_feat
    store_mask = (batch_offsets[:, None] < batch_size) & (output_offsets[None, :] < output_size)
    tl.store(o_ptrs, acc, mask=store_mask)

def fused_i2h_tanh_optimized(x, h_prev, weight_i2h, bias_i2h):
    batch_size, input_size = x.shape
    hidden_size = weight_i2h.shape[0]
    
    h_next = torch.empty((batch_size, hidden_size), device=x.device, dtype=x.dtype)
    
    # Auto-tune for register pressure
    configs = [
        {'BLOCK_BATCH': 16, 'BLOCK_HIDDEN': 64, 'BLOCK_FEAT': 64, 'USE_TF32': True},
        {'BLOCK_BATCH': 32, 'BLOCK_HIDDEN': 32, 'BLOCK_FEAT': 64, 'USE_TF32': True},
        {'BLOCK_BATCH': 16, 'BLOCK_HIDDEN': 32, 'BLOCK_FEAT': 128, 'USE_TF32': True},
    ]
    
    grid = lambda META: (
        triton.cdiv(batch_size, META['BLOCK_BATCH']),
        triton.cdiv(hidden_size, META['BLOCK_HIDDEN'])
    )
    
    # Use smallest config first (lowest register pressure)
    fused_i2h_tanh_kernel_optimized[grid](
        x, h_prev, weight_i2h, bias_i2h, h_next,
        batch_size, input_size, hidden_size,
        x.stride(0), x.stride(1),
        h_prev.stride(0), h_prev.stride(1),
        weight_i2h.stride(0), weight_i2h.stride(1),
        bias_i2h.stride(0),
        **configs[1]  # Middle ground
    )
    
    return h_next

def fused_h2o_optimized(h, weight_h2o, bias_h2o):
    batch_size, hidden_size = h.shape
    output_size = weight_h2o.shape[0]
    
    output = torch.empty((batch_size, output_size), device=h.device, dtype=h.dtype)
    
    # Auto-tune configs
    configs = [
        {'BLOCK_BATCH': 16, 'BLOCK_OUTPUT': 64, 'BLOCK_HIDDEN': 64, 'USE_TF32': True},
        {'BLOCK_BATCH': 32, 'BLOCK_OUTPUT': 32, 'BLOCK_HIDDEN': 64, 'USE_TF32': True},
        {'BLOCK_BATCH': 16, 'BLOCK_OUTPUT': 32, 'BLOCK_HIDDEN': 128, 'USE_TF32': True},
    ]
    
    grid = lambda META: (
        triton.cdiv(batch_size, META['BLOCK_BATCH']),
        triton.cdiv(output_size, META['BLOCK_OUTPUT'])
    )
    
    fused_h2o_kernel_optimized[grid](
        h, weight_h2o, bias_h2o, output,
        batch_size, hidden_size, output_size,
        h.stride(0), h.stride(1),
        weight_h2o.stride(0), weight_h2o.stride(1),
        bias_h2o.stride(0),
        output.stride(0), output.stride(1),
        **configs[1]  # Middle ground
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weight_i2h = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size))
        self.bias_i2h = nn.Parameter(torch.randn(hidden_size))
        self.weight_h2o = nn.Parameter(torch.randn(output_size, hidden_size))
        self.bias_h2o = nn.Parameter(torch.randn(output_size))
        
        # Initialize weights for better convergence
        nn.init.kaiming_normal_(self.weight_i2h, mode='fan_in', nonlinearity='tanh')
        nn.init.kaiming_normal_(self.weight_h2o, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.bias_i2h)
        nn.init.zeros_(self.bias_h2o)
    
    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]
            hidden = fused_i2h_tanh_optimized(x_t, hidden, self.weight_i2h, self.bias_i2h)
            output_t = fused_h2o_optimized(hidden, self.weight_h2o, self.bias_h2o)
            outputs.append(output_t)
        
        return torch.stack(outputs, dim=0)
