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
):
    # Optimized grid: 2D for matrix multiplication output
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # hidden dimension
    
    # Shared offsets for fused operations
    batch_offs = pid_m * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    hidden_offs = pid_n * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    
    # Initialize accumulator for matmul
    acc = tl.zeros((BLOCK_BATCH, BLOCK_HIDDEN), dtype=tl.float32)
    
    # Optimized inner loop with prefetching
    total_features = input_size + hidden_size
    for k in range(0, total_features, BLOCK_FEAT):
        k_offs = k + tl.arange(0, BLOCK_FEAT)
        
        # Load weight tile - shared by all batch elements
        w_mask = (hidden_offs[:, None] < hidden_size) & (k_offs[None, :] < total_features)
        w_ptrs = w_i2h_ptr + hidden_offs[:, None] * stride_w_hidden + k_offs[None, :] * stride_w_feat
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Fused input/hidden loading with optimized memory access
        # Create combined feature vector [x, h_prev]
        if k < input_size:
            # Load input features
            input_k_offs = k_offs
            input_mask = (batch_offs[:, None] < batch_size) & (input_k_offs[None, :] < input_size)
            x_ptrs = x_ptr + batch_offs[:, None] * stride_x_batch + input_k_offs[None, :] * stride_x_feat
            data_block = tl.load(x_ptrs, mask=input_mask, other=0.0)
        else:
            # Load hidden features
            hidden_k_offs = k_offs - input_size
            hidden_mask = (batch_offs[:, None] < batch_size) & (hidden_k_offs[None, :] < hidden_size)
            h_ptrs = h_prev_ptr + batch_offs[:, None] * stride_h_batch + hidden_k_offs[None, :] * stride_h_feat
            data_block = tl.load(h_ptrs, mask=hidden_mask, other=0.0)
        
        # Accumulate matrix multiplication
        acc += tl.dot(data_block, w_block.T, allow_tf32=True)
    
    # Fused bias addition - using same offsets as matmul
    bias = tl.load(b_i2h_ptr + hidden_offs, mask=hidden_offs < hidden_size, other=0.0)
    acc += bias[None, :]
    
    # Fused tanh activation with stable computation
    # Clip to prevent overflow in exp
    x = tl.minimum(tl.maximum(acc, -20.0), 20.0)
    exp_2x = tl.exp(2.0 * x)
    tanh_result = (exp_2x - 1.0) / (exp_2x + 1.0)
    
    # Store with same offsets as computation
    h_next_ptrs = h_next_ptr + batch_offs[:, None] * stride_h_batch + hidden_offs[None, :] * stride_h_feat
    store_mask = (batch_offs[:, None] < batch_size) & (hidden_offs[None, :] < hidden_size)
    tl.store(h_next_ptrs, tanh_result, mask=store_mask)

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
    # Optimized grid: 2D for output matrix
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # output dimension
    
    # Shared offsets for fused operations
    batch_offs = pid_m * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
    output_offs = pid_n * BLOCK_OUTPUT + tl.arange(0, BLOCK_OUTPUT)
    
    # Initialize accumulator for matmul
    acc = tl.zeros((BLOCK_BATCH, BLOCK_OUTPUT), dtype=tl.float32)
    
    # Optimized inner loop for matrix multiplication
    for k in range(0, hidden_size, BLOCK_HIDDEN):
        k_offs = k + tl.arange(0, BLOCK_HIDDEN)
        
        # Load hidden state tile
        h_mask = (batch_offs[:, None] < batch_size) & (k_offs[None, :] < hidden_size)
        h_ptrs = h_ptr + batch_offs[:, None] * stride_h_batch + k_offs[None, :] * stride_h_feat
        h_block = tl.load(h_ptrs, mask=h_mask, other=0.0)
        
        # Load weight tile
        w_mask = (output_offs[:, None] < output_size) & (k_offs[None, :] < hidden_size)
        w_ptrs = w_h2o_ptr + output_offs[:, None] * stride_w_output + k_offs[None, :] * stride_w_feat
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Accumulate matrix multiplication
        acc += tl.dot(h_block, w_block.T, allow_tf32=True)
    
    # Fused bias addition - using same offsets
    bias = tl.load(b_h2o_ptr + output_offs, mask=output_offs < output_size, other=0.0)
    acc += bias[None, :]
    
    # Store with same offsets as computation (linear activation)
    o_ptrs = o_ptr + batch_offs[:, None] * stride_o_batch + output_offs[None, :] * stride_o_feat
    store_mask = (batch_offs[:, None] < batch_size) & (output_offs[None, :] < output_size)
    tl.store(o_ptrs, acc, mask=store_mask)

def fused_i2h_tanh(x, h_prev, weight_i2h, bias_i2h):
    batch_size, input_size = x.shape
    hidden_size = weight_i2h.shape[0]
    
    h_next = torch.empty((batch_size, hidden_size), device=x.device, dtype=x.dtype)
    
    # Optimized grid calculation for better occupancy
    grid = lambda META: (
        triton.cdiv(batch_size, META['BLOCK_BATCH']),
        triton.cdiv(hidden_size, META['BLOCK_HIDDEN'])
    )
    
    # Tuned block sizes for Ada Lovelace architecture
    fused_i2h_tanh_kernel[grid](
        x, h_prev, weight_i2h, bias_i2h, h_next,
        batch_size, input_size, hidden_size,
        x.stride(0), x.stride(1),
        h_prev.stride(0), h_prev.stride(1),
        weight_i2h.stride(0), weight_i2h.stride(1),
        bias_i2h.stride(0),
        BLOCK_BATCH=32, BLOCK_HIDDEN=128, BLOCK_FEAT=64  # Optimized for tensor cores
    )
    
    return h_next

def fused_h2o(h, weight_h2o, bias_h2o):
    batch_size, hidden_size = h.shape
    output_size = weight_h2o.shape[0]
    
    output = torch.empty((batch_size, output_size), device=h.device, dtype=h.dtype)
    
    # Optimized grid calculation for better occupancy
    grid = lambda META: (
        triton.cdiv(batch_size, META['BLOCK_BATCH']),
        triton.cdiv(output_size, META['BLOCK_OUTPUT'])
    )
    
    # Tuned block sizes for Ada Lovelace architecture
    fused_h2o_kernel[grid](
        h, weight_h2o, bias_h2o, output,
        batch_size, hidden_size, output_size,
        h.stride(0), h.stride(1),
        weight_h2o.stride(0), weight_h2o.stride(1),
        bias_h2o.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_BATCH=32, BLOCK_OUTPUT=128, BLOCK_HIDDEN=64  # Optimized for tensor cores
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
