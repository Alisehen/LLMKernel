import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def argmin_lastdim_kernel(
    x_ptr,          # *f32 / *f16 / *bf16 etc.
    out_ptr,        # *i64
    M,              # number of rows (outer elements)
    K,              # reduction length (size of last dim)
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row = pid
    row_mask = row < M

    offs_k = tl.arange(0, BLOCK_K)

    # Initialize best values and indices
    best_vals = tl.full((BLOCK_K,), float("inf"), tl.float32)
    best_indices = tl.full((BLOCK_K,), 0, tl.int32)

    # Iterate over K dimension in chunks of BLOCK_K
    k = 0
    while k < K:
        idx = k + offs_k
        mask = (idx < K) & row_mask

        # Flat row-major offset: row * K + idx
        ptrs = x_ptr + row * K + idx
        vals = tl.load(ptrs, mask=mask, other=float("inf"))
        vals_f32 = vals.to(tl.float32)

        is_better = vals_f32 < best_vals
        best_vals = tl.where(is_better, vals_f32, best_vals)
        best_indices = tl.where(is_better, idx, best_indices)

        k += BLOCK_K

    # Reduce across BLOCK_K lanes to get final argmin index for this row
    min_val = tl.min(best_vals, axis=0)
    is_min = best_vals == min_val
    # For lanes that are not minima, set index to a large sentinel (K)
    idx_candidates = tl.where(is_min, best_indices, K)
    final_idx = tl.min(idx_candidates, axis=0)

    # Store result (only for valid rows)
    final_idx_i64 = tl.cast(final_idx, tl.int64)
    tl.store(out_ptr + row, final_idx_i64, mask=row_mask)


def triton_argmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Fallback to PyTorch on non-CUDA tensors to ensure correctness
    if not x.is_cuda:
        return torch.argmin(x, dim=dim)

    ndim = x.ndim
    if dim < 0:
        dim += ndim

    # Move reduction dim to last for contiguous, strided-friendly layout
    x_moved = x.movedim(dim, -1).contiguous()
    *outer_shape, K = x_moved.shape
    M = int(torch.tensor(outer_shape).prod().item()) if outer_shape else 1

    x_2d = x_moved.view(M, K)
    out_flat = torch.empty(M, dtype=torch.int64, device=x.device)

    BLOCK_K = 128  # power-of-two block size

    # Grid: one program per row; ensure > 0 as required
    grid = lambda meta: (max(1, M),)

    argmin_lastdim_kernel[grid](
        x_2d,
        out_flat,
        M,
        K,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    out = out_flat.view(*outer_shape)
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized model that finds the index of the minimum value along a specified dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_argmin(x, self.dim)
