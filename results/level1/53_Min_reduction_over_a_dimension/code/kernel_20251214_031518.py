import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def min_reduce_3d_kernel(
    x_ptr,
    out_ptr,
    B, M, N,
    strideB, strideM, strideN,
    n_out,
    reduce_size,
    BLOCK_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIM: tl.constexpr,
):
    """
    Reduce a 3D tensor x[b, m, n] along a given dimension DIM (0, 1, or 2)
    and write the minima into a flattened 1D output tensor.

    Layout assumptions:
      - x_ptr points to a 3D tensor with shape (B, M, N) and strides
        (strideB, strideM, strideN) in elements.
      - out_ptr is a 1D tensor with n_out elements, laid out contiguously.
      - n_out is:
          DIM = 0 -> M * N (output shape (M, N))
          DIM = 1 -> B * N (output shape (B, N))
          DIM = 2 -> B * M (output shape (B, M))
      - reduce_size is:
          DIM = 0 -> B
          DIM = 1 -> M
          DIM = 2 -> N
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_OUT
    offs = block_start + tl.arange(0, BLOCK_OUT)
    mask_out = offs < n_out

    # Map flattened output index -> (b, m, n) indices of non-reduced dims
    # according to DIM. This branch is decided at compile-time.
    if DIM == 0:
        # Reduce over B, output shape (M, N), row-major flatten
        # offs in [0, M*N): m = offs // N, n = offs % N
        m_idx = offs // N
        n_idx = offs % N
    elif DIM == 1:
        # Reduce over M, output shape (B, N), row-major flatten
        # offs in [0, B*N): b = offs // N, n = offs % N
        b_idx = offs // N
        n_idx = offs % N
    else:
        # DIM == 2
        # Reduce over N, output shape (B, M), row-major flatten
        # offs in [0, B*M): b = offs // M, m = offs % M
        b_idx = offs // M
        m_idx = offs % M

    # Initialize running minimum with +inf
    min_val = tl.full([BLOCK_OUT], float('inf'), dtype=tl.float32)

    # Loop over the reduced dimension in tiles of size BLOCK_K
    for k in range(0, reduce_size, BLOCK_K):
        k_idx = k + tl.arange(0, BLOCK_K)
        mask_k = k_idx < reduce_size

        # Build pointers for a [BLOCK_K, BLOCK_OUT] tile
        if DIM == 0:
            # k dimension is B
            b_tile = k_idx
            ptrs = (
                x_ptr
                + b_tile[:, None] * strideB
                + m_idx[None, :] * strideM
                + n_idx[None, :] * strideN
            )
        elif DIM == 1:
            # k dimension is M
            m_tile = k_idx
            ptrs = (
                x_ptr
                + b_idx[None, :] * strideB
                + m_tile[:, None] * strideM
                + n_idx[None, :] * strideN
            )
        else:
            # DIM == 2, k dimension is N
            n_tile = k_idx
            ptrs = (
                x_ptr
                + b_idx[None, :] * strideB
                + m_idx[None, :] * strideM
                + n_tile[:, None] * strideN
            )

        mask = mask_k[:, None] & mask_out[None, :]
        vals = tl.load(ptrs, mask=mask, other=float('inf'))

        # Reduce along the K axis, then update running minimum
        tile_min = tl.min(vals, axis=0)
        min_val = tl.minimum(min_val, tile_min)

    # Store the result
    tl.store(out_ptr + offs, min_val, mask=mask_out)


def triton_min_reduce_3d(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Triton-based min reduction over a specified dimension for 3D tensors.

    Falls back to torch.min for unsupported cases (non-CUDA, non-float32, non-3D).
    """
    if x.dim() != 3 or not x.is_cuda or x.dtype != torch.float32:
        # Fallback for generality / CPU / non-fp32
        return torch.min(x, dim=dim)[0]

    dim = dim % x.dim()  # handle negative dims

    B, M, N = x.shape
    strideB, strideM, strideN = x.stride()

    if dim == 0:
        out_shape = (M, N)
        n_out = M * N
        reduce_size = B
        DIM = 0
    elif dim == 1:
        out_shape = (B, N)
        n_out = B * N
        reduce_size = M
        DIM = 1
    else:
        # dim == 2
        out_shape = (B, M)
        n_out = B * M
        reduce_size = N
        DIM = 2

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    BLOCK_OUT = 128  # power of 2
    BLOCK_K = 128    # power of 2

    grid = lambda meta: (triton.cdiv(n_out, meta["BLOCK_OUT"]),)

    min_reduce_3d_kernel[grid](
        x,
        out,
        B, M, N,
        strideB, strideM, strideN,
        n_out,
        reduce_size,
        BLOCK_OUT=BLOCK_OUT,
        BLOCK_K=BLOCK_K,
        DIM=DIM,
    )

    return out


class ModelNew(nn.Module):
    """
    Simple model that performs min reduction over a specific dimension,
    accelerated with a Triton kernel for 3D CUDA float32 inputs.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_min_reduce_3d(x, self.dim)
