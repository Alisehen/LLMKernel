import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def max_reduce_strided_kernel(
    x_ptr,         # *f32 / *f16 / *bf16
    out_ptr,       # *f32 / *f16 / *bf16
    M,             # number of output elements (rows)
    K,             # reduction size (size along reduced dimension)

    # Strides of x (in elements) for up to 8 dimensions
    stride_x0, stride_x1, stride_x2, stride_x3,
    stride_x4, stride_x5, stride_x6, stride_x7,

    # Sizes of x for up to 8 dimensions
    size_x0, size_x1, size_x2, size_x3,
    size_x4, size_x5, size_x6, size_x7,

    stride_out,      # stride of out (in elements) along its single logical dimension
    stride_reduce,   # stride of x along the reduction dimension (in elements)

    REDUCE_DIM: tl.constexpr,  # which dimension to reduce over (0-based)
    BLOCK_SIZE: tl.constexpr,  # tile size along K (power-of-2)
):
    # One program per output element (i.e., per combination of all dims except REDUCE_DIM)
    pid = tl.program_id(axis=0)
    m = pid

    is_valid_row = m < M

    # ---- Map linear row index m to multi-dimensional indices (excluding REDUCE_DIM) ----
    # We enumerate all non-reduced dimensions in row-major order (last dim fastest) using
    # the original dimension order with REDUCE_DIM removed. This matches PyTorch's
    # flattening behavior for the output tensor.
    tmp = m
    offset = 0

    # Helper macros (compile-time) to select size/stride for a given dimension d
    def get_size(d):
        if d == 0:
            return size_x0
        elif d == 1:
            return size_x1
        elif d == 2:
            return size_x2
        elif d == 3:
            return size_x3
        elif d == 4:
            return size_x4
        elif d == 5:
            return size_x5
        elif d == 6:
            return size_x6
        else:
            return size_x7

    def get_stride(d):
        if d == 0:
            return stride_x0
        elif d == 1:
            return stride_x1
        elif d == 2:
            return stride_x2
        elif d == 3:
            return stride_x3
        elif d == 4:
            return stride_x4
        elif d == 5:
            return stride_x5
        elif d == 6:
            return stride_x6
        else:
            return stride_x7

    MAX_DIMS = 8  # must match host-side packing

    # Process dimensions AFTER the reduction dim: (MAX_DIMS-1 .. REDUCE_DIM+1)
    for d in range(MAX_DIMS - 1, REDUCE_DIM, -1):
        size_d = get_size(d)
        stride_d = get_stride(d)
        # size_d may be 1 for padding dims; modulo/division are safe
        idx_d = tmp % size_d
        tmp = tmp // size_d
        offset += idx_d * stride_d

    # Process dimensions BEFORE the reduction dim: (REDUCE_DIM-1 .. 0)
    for d in range(REDUCE_DIM - 1, -1, -1):
        size_d = get_size(d)
        stride_d = get_stride(d)
        idx_d = tmp % size_d
        tmp = tmp // size_d
        offset += idx_d * stride_d

    # Base pointer for this logical row (all dims fixed except REDUCE_DIM)
    row_base_ptr = x_ptr + offset

    # ---- Reduction over K elements along REDUCE_DIM with stride stride_reduce ----
    running_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)

    for k in range(0, K, BLOCK_SIZE):
        offs_k = k + tl.arange(0, BLOCK_SIZE)
        mask_k = (offs_k < K) & is_valid_row
        ptrs = row_base_ptr + offs_k * stride_reduce
        vals = tl.load(ptrs, mask=mask_k, other=-float("inf"))
        vals_f32 = vals.to(tl.float32)
        running_max = tl.maximum(running_max, vals_f32)

    max_val = tl.max(running_max, axis=0)

    # Store result for this row; out is viewed as 1D of length M
    out_ptr_m = out_ptr + m * stride_out
    tl.store(out_ptr_m, max_val, mask=is_valid_row)


def triton_max_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Max reduction over a given dimension using a high-performance Triton kernel.
    Returns only the values (same as torch.max(x, dim=dim)[0]), with the same
    output shape and dtype as PyTorch.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), \
        "Supported dtypes: fp16, bf16, fp32"

    ndim = x.ndim
    assert ndim > 0, "Input must have at least one dimension"

    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim, "Invalid reduction dimension"

    # Size of the reduction dimension
    K = x.shape[dim]
    assert K > 0, "torch.max does not support reduction over an empty dimension"

    # Number of output elements = product of all sizes except along dim
    M = x.numel() // K

    # Output shape: original shape with the reduction dimension removed
    out_shape = x.shape[:dim] + x.shape[dim + 1 :]
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    out_flat = out.view(-1)
    stride_out = out_flat.stride(0)

    # Prepare sizes and strides for up to MAX_DIMS = 8
    MAX_DIMS = 8
    assert ndim <= MAX_DIMS, f"Only tensors with up to {MAX_DIMS} dims are supported"

    sizes = list(x.shape) + [1] * (MAX_DIMS - ndim)
    strides = list(x.stride()) + [0] * (MAX_DIMS - ndim)

    stride_reduce = x.stride(dim)

    # Kernel launch configuration
    BLOCK_SIZE = 128  # power-of-2 tile along K

    grid = lambda meta: (max(M, 1),)

    max_reduce_strided_kernel[grid](
        x,
        out_flat,
        M,
        K,
        strides[0], strides[1], strides[2], strides[3],
        strides[4], strides[5], strides[6], strides[7],
        sizes[0], sizes[1], sizes[2], sizes[3],
        sizes[4], sizes[5], sizes[6], sizes[7],
        stride_out,
        stride_reduce,
        REDUCE_DIM=dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return out


class ModelNew(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension
    using a high-performance, stride-aware Triton kernel.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_max_dim(x, self.dim)
