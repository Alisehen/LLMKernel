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
    BLOCK_BATCH: tl.constexpr, BLOCK_HIDDEN: tl.constexpr, BLOCK_INPUT: tl.constexpr, BLOCK_HIDDEN_PREV: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_hidden = tl.program_id(1)
    
    batch_offsets = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    hidden_offsets = pid_hidden * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    
    acc = tl.zeros((BLOCK_BATCH, BLOCK_HIDDEN), dtype=tl.float32)
    
    # Loop over input features
    for k in range(0, input_size, BLOCK_INPUT):
        k_offsets = k + tl.arange(0, BLOCK_INPUT)
        
        # Load input block
        x_mask = (batch_offsets[:, None] < batch_size) & (k_offsets[None, :] < input_size)
        x_ptrs = x_ptr + batch_offsets[:, None] * stride_x_batch + k_offsets[None, :] * stride_x_feat
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load weight block for input part
        w_mask = (hidden_offsets[:, None] < hidden_size) & (k_offsets[None, :] < input_size)
        w_ptrs = w_i2h_ptr + hidden_offsets[:, None] * stride_w_hidden + k_offsets[None, :] * stride_w_feat
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc += tl.dot(x_block, w_block, allow_tf32=True)
    
    # Loop over previous hidden features
    for k in range(0, hidden_size, BLOCK_HIDDEN_PREV):
        k_offsets = k + tl.arange(0, BLOCK_HIDDEN_PREV)
        
        # Load previous hidden block
        h_mask = (batch_offsets[:, None] < batch_size) & (k_offsets[None, :] < hidden_size)
        h_ptrs = h_prev_ptr + batch_offsets[:, None] * stride_h_batch + k_offsets[None, :] * stride_h_feat
        h_block = tl.load(h_ptrs, mask=h_mask, other=0.0)
        
        # Load weight block for hidden part (offset by input_size)
        w_mask = (hidden_offsets[:, None] < hidden_size) & (k_offsets[None, :] < hidden_size)
        w_ptrs = w_i2h_ptr + hidden_offsets[:, None] * stride_w_hidden + (input_size + k_offsets[None, :]) * stride_w_feat
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc += tl.dot(h_block, w_block, allow_tf32=True)
    
    # Add bias
    bias = tl.load(b_i2h_ptr + hidden_offsets, mask=hidden_offsets < hidden_size, other=0.0)
    acc += bias[None, :]
    
    # FIXED: Manual tanh implementation - Triton doesn't have tl.tanh
    # For moderate values: use (exp(2*x) - 1)/(exp(2*x) + 1)
    # For large positive values: tanh(x) ≈ 1 - 2*exp(-2*x)
    # For large negative values: tanh(x) ≈ -1 + 2*exp(2*x)
    
    # Compute absolute value for clamping decisions
    abs_acc = tl.abs(acc)
    
    # For positive values, use exp(-2*x) when x > 20 to prevent overflow
    # For negative values, use exp(2*x) when x < -20 to prevent underflow
    positive_large = acc > 20.0
    negative_large = acc < -20.0
    moderate = ~(positive_large | negative_large)
    
    # Initialize result
    result = tl.zeros_like(acc)
    
    # For moderate values: use standard formula
    # Compute exp(2*acc) only for moderate values to avoid unnecessary computation
    moderate_acc = tl.where(moderate, acc, 0.0)
    exp_2x_moderate = tl.exp(2.0 * moderate_acc)
    moderate_result = (exp_2x_moderate - 1.0) / (exp_2x_moderate + 1.0)
    result = tl.where(moderate, moderate_result, result)
    
    # For large positive values: tanh(x) ≈ 1 - 2*exp(-2*x) for x > 20
    positive_acc = tl.where(positive_large, acc, 0.0)
    exp_neg = tl.exp(-2.0 * positive_acc)
    positive_result = 1.0 - 2.0 * exp_neg / (1.0 + exp_neg)
    result = tl.where(positive_large, positive_result, result)
    
    # For large negative values: tanh(x) ≈ -1 + 2*exp(2*x) for x < -20
    negative_acc = tl.where(negative_large, acc, 0.0)
    exp_pos = tl.exp(2.0 * negative_acc)
    negative_result = -1.0 + 2.0 * exp_pos / (1.0 + exp_pos)
    result = tl.where(negative_large, negative_result, result)
    
    # Store new hidden state
    h_next_ptrs = h_next_ptr + batch_offsets[:, None] * stride_h_batch + hidden_offsets[None, :] * stride_h_feat
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
):
    pid_batch = tl.program_id(0)
    pid_output = tl.program_id(1)
    
    batch_offsets = pid_batch * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    output_offsets = pid_output * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
    
    acc = tl.zeros((BLOCK_BATCH, BLOCK_OUTPUT), dtype=tl.float32)
    
    # Loop over hidden features
    for k in range(0, hidden_size, BLOCK_HIDDEN):
        k_offsets = k + tl.arange(0, BLOCK_HIDDEN)
        
        # Load hidden block
        h_mask = (batch_offsets[:, None] < batch_size) & (k_offsets[None, :] < hidden_size)
        h_ptrs = h_ptr + batch_offsets[:, None] * stride_h_batch + k_offsets[None, :] * stride_h_feat
        h_block = tl.load(h_ptrs, mask=h_mask, other=0.0)
        
        # Load weight block
        w_mask = (output_offsets[:, None] < output_size) & (k_offsets[None, :] < hidden_size)
        w_ptrs = w_h2o_ptr + output_offsets[:, None] * stride_w_output + k_offsets[None, :] * stride_w_feat
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc += tl.dot(h_block, w_block, allow_tf32=True)
    
    # Add bias
    bias = tl.load(b_h2o_ptr + output_offsets, mask=output_offsets < output_size, other=0.0)
    acc += bias[None, :]
    
    # Store output
    o_ptrs = o_ptr + batch_offsets[:, None] * stride_o_batch + output_offsets[None, :] * stride_o_feat
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
    
    fused_i2h_tanh_kernel[grid](
        x, h_prev, weight_i2h, bias_i2h, h_next,
        batch_size, input_size, hidden_size,
        x.stride(0), x.stride(1),
        h_prev.stride(0), h_prev.stride(1),
        weight_i2h.stride(0), weight_i2h.stride(1),
        bias_i2h.stride(0),
        BLOCK_BATCH=32, BLOCK_HIDDEN=64, BLOCK_INPUT=64, BLOCK_HIDDEN_PREV=64
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
        BLOCK_BATCH=32, BLOCK_OUTPUT=64, BLOCK_HIDDEN=64
    )
    
    return output

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize parameters
        self.weight_i2h = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size))
        self.bias_i2h = nn.Parameter(torch.randn(hidden_size))
        self.weight_h2o = nn.Parameter(torch.randn(output_size, hidden_size))
        self.bias_h2o = nn.Parameter(torch.randn(output_size))
    
    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, _ = x.size()
        hidden = h0.to(x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]  # (batch_size, input_size)
            hidden = fused_i2h_tanh(x_t, hidden, self.weight_i2h, self.bias_i2h)
            output_t = fused_h2o(hidden, self.weight_h2o, self.bias_h2o)
            outputs.append(output_t)
        
        return torch.stack(outputs, dim=0)
