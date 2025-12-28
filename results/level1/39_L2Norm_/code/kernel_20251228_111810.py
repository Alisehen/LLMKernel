import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def l2norm_sumsq_kernel(
    x_ptr,
    norm_ptr,
    B,
    D,
    stride_xm,
    stride_xn,
    stride_nm,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row = pid

    # Vectorized row mask to satisfy "mask shape must match data shape"
    offs_n = tl.arange(0, BLOCK_N)
    row_ids = row + tl.zeros((BLOCK_N,), dtype=tl.int32)
    valid_row = row_ids < B

    row_x_ptr = x_ptr + row * stride_xm
    row_norm_ptr = norm_ptr + row * stride_nm

    sum_sq = tl.zeros((1,), dtype=tl.float32)

    col_start = 0
    while col_start < D:
        cols = col_start + offs_n
        col_mask = cols < D
        mask = col_mask & valid_row

        x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        tile_sum = tl.sum(x_f32 * x_f32, axis=0)
        sum_sq += tile_sum
        col_start += BLOCK_N

    norm = tl.sqrt(sum_sq)

    norm_offs = tl.arange(0, 1)
    norm_mask = (row + norm_offs) < B
    tl.store(row_norm_ptr + norm_offs, norm, mask=norm_mask)


@triton.jit
def l2norm_normalize_kernel(
    x_ptr,
    norm_ptr,
    y_ptr,
    B,
    D,
    stride_xm,
    stride_xn,
    stride_nm,
    stride_nn,
    stride_ym,
    stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row = pid

    offs_n = tl.arange(0, BLOCK_N)
    row_ids = row + tl.zeros((BLOCK_N,), dtype=tl.int32)
    valid_row = row_ids < B

    row_x_ptr = x_ptr + row * stride_xm
    row_y_ptr = y_ptr + row * stride_ym

    # Load per-row norm (shape (1,)) and broadcast
    norm_offs = tl.arange(0, 1)
    norm_mask = (row + norm_offs) < B
    norm = tl.load(norm_ptr + row * stride_nm + norm_offs * stride_nn, mask=norm_mask, other=1.0)
    norm_block = tl.broadcast_to(norm, offs_n.shape)

    col_start = 0
    while col_start < D:
        cols = col_start + offs_n
        col_mask = cols < D
        mask = col_mask & valid_row

        x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0)
        y = x / norm_block
        tl.store(row_y_ptr + cols * stride_yn, y, mask=mask)
        col_start += BLOCK_N


def triton_l2norm(x: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize along dim=1 using Triton.
    Expects 2D input [B, D] on CUDA.
    """
    assert x.ndim == 2, "Only 2D tensors are supported"
    B, D = x.shape
    if B == 0:
        return x.clone()

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    # norms has shape [B, 1] to match keepdim=True behavior
    norms = torch.empty((B, 1), device=x.device, dtype=torch.float32)

    stride_xm, stride_xn = x_contig.stride()
    stride_ym, stride_yn = y.stride()
    stride_nm, stride_nn = norms.stride()

    BLOCK_N = 256

    grid = lambda meta: (max(1, B),)

    l2norm_sumsq_kernel[grid](
        x_contig,
        norms,
        B,
        D,
        stride_xm,
        stride_xn,
        stride_nm,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    l2norm_normalize_kernel[grid](
        x_contig,
        norms,
        y,
        B,
        D,
        stride_xm,
        stride_xn,
        stride_nm,
        stride_nn,
        stride_ym,
        stride_yn,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated L2 normalization along dim=1.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch on CPU; use Triton on CUDA
        if not x.is_cuda:
            return x / torch.norm(x, p=2, dim=1, keepdim=True)
        return triton_l2norm(x)
