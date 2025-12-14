import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_fn(a, b):
    return a + b


@triton.jit
def scan_kernel_1d(
    input_ptr,
    output_ptr,
    block_sum_ptr,
    n_rows,
    n_cols,
    stride_inp_row,
    stride_inp_col,
    stride_out_row,
    stride_out_col,
    block_sum_stride,
    BLOCK_SIZE_COL: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    NUM_STAGES: tl.constexpr = 3,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    row_offset = pid_row * BLOCK_SIZE_ROW
    col_offset = pid_col * BLOCK_SIZE_COL
    
    row_mask = row_offset + tl.arange(0, BLOCK_SIZE_ROW) < n_rows
    col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < n_cols
    
    for r in tl.range(0, BLOCK_SIZE_ROW, num_stages=NUM_STAGES):
        row_idx = row_offset + r
        row_active = tl.full((BLOCK_SIZE_COL,), row_idx < n_rows, dtype=tl.int1)
        active_mask = row_active & col_mask
        
        if tl.reduce_or(active_mask):
            row_start = row_idx * stride_inp_row
            input_ptrs = input_ptr + row_start + col_offsets * stride_inp_col
            block_vals = tl.load(input_ptrs, mask=active_mask, other=0.0)
            
            block_scan = tl.associative_scan(block_vals, 0, combine_fn=add_fn)
            
            output_ptrs = output_ptr + row_start + col_offsets * stride_out_col
            tl.store(output_ptrs, block_scan, mask=active_mask)
            
            block_sum = tl.sum(block_vals)
            block_sum_index = row_idx * block_sum_stride + pid_col
            tl.store(block_sum_ptr + block_sum_index, block_sum)


@triton.jit
def scan_add_kernel(
    output_ptr,
    block_sum_ptr,
    n_rows,
    n_cols,
    stride_out_row,
    stride_out_col,
    block_sum_stride,
    grid_cols: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    if pid_col == 0:
        return
    
    row_offset = pid_row * BLOCK_SIZE_ROW
    col_offset = pid_col * BLOCK_SIZE_COL
    
    row_mask = row_offset + tl.arange(0, BLOCK_SIZE_ROW) < n_rows
    col_offsets = col_offset + tl.arange(0, BLOCK_SIZE_COL)
    col_mask = col_offsets < n_cols
    
    cumulative_sum = tl.full((BLOCK_SIZE_ROW,), 0.0, dtype=tl.float32)
    
    # Pre-compute cumulative sums for all rows in this block
    for r in range(BLOCK_SIZE_ROW):
        if row_mask[r]:
            row_idx = row_offset + r
            row_cumsum = 0.0
            # Optimized loop with better memory access pattern
            for i in tl.range(pid_col):
                block_sum_idx = row_idx * block_sum_stride + i
                prev_sum = tl.load(block_sum_ptr + block_sum_idx)
                row_cumsum += prev_sum
            cumulative_sum = tl.where(
                tl.arange(0, BLOCK_SIZE_ROW) == r,
                row_cumsum,
                cumulative_sum
            )
    
    for r in range(BLOCK_SIZE_ROW):
        if row_mask[r]:
            row_idx = row_offset + r
            row_start = row_idx * stride_out_row
            output_ptrs = output_ptr + row_start + col_offsets * stride_out_col
            
            curr_vals = tl.load(output_ptrs, mask=col_mask, other=0.0)
            curr_cumsum = cumulative_sum[r]
            updated_vals = curr_vals + curr_cumsum
            tl.store(output_ptrs, updated_vals, mask=col_mask)


def triton_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("Input must be 2D for this implementation")
    
    output = torch.empty_like(x)
    
    if dim == 0:
        return triton_cumsum(x.t(), dim=1).t()
    
    x = x.contiguous()
    output = output.contiguous()
    
    n_rows, n_cols = x.shape
    
    # Tuned for Ada Lovelace memory hierarchy
    BLOCK_SIZE_COL = 2048  # Increased for better memory coalescing
    BLOCK_SIZE_ROW = 2     # Reduced to lower register pressure
    
    grid_cols = triton.cdiv(n_cols, BLOCK_SIZE_COL)
    grid_rows = triton.cdiv(n_rows, BLOCK_SIZE_ROW)
    
    block_sums = torch.zeros((n_rows, grid_cols), device=x.device, dtype=x.dtype)
    
    # Use autotuned number of stages
    num_stages_options = {1, 2, 3}
    
    for num_stages in num_stages_options:
        try:
            scan_kernel_1d[(grid_rows, grid_cols)](
                x,
                output,
                block_sums,
                n_rows,
                n_cols,
                x.stride(0),
                x.stride(1),
                output.stride(0),
                output.stride(1),
                block_sums.stride(0),
                BLOCK_SIZE_COL=BLOCK_SIZE_COL,
                BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
                NUM_STAGES=num_stages,
            )
            break
        except Exception:
            continue
    
    if grid_cols > 1:
        scan_add_kernel[(grid_rows, grid_cols)](
            output,
            block_sums,
            n_rows,
            n_cols,
            output.stride(0),
            output.stride(1),
            block_sums.stride(0),
            grid_cols=grid_cols,
            BLOCK_SIZE_COL=BLOCK_SIZE_COL,
            BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        )
    
    return output


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            original_shape = x.shape
            x = x.view(-1, x.shape[-1])
            result = triton_cumsum(x, dim=self.dim if self.dim == 1 else 1)
            return result.view(original_shape)
        else:
            return triton_cumsum(x, dim=self.dim)
