import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def l2norm_fused_kernel(
    x_ptr,        # *const T
    y_ptr,        # *mut T
    B,            # int32: number of rows
    D,            # int32: number of cols
    stride_xm,    # int32
    stride_xn,    # int32
    stride_ym,    # int32
    stride_yn,    # int32
    BLOCK_N: tl.constexpr,
):
    """
    Fused L2 normalization kernel:
      - First pass: compute sum of squares along dim=1 (per row)
      - Second pass: normalize each row by its L2 norm
    One program_id handles one row.
    """
    pid = tl.program_id(axis=0)
    row = pid

    # Scalar mask for row validity
    row_mask = row < B

    # Offsets along the feature dim
    offs_n = tl.arange(0, BLOCK_N)

    # Base pointers for this row
    row_x_ptr = x_ptr + row * stride_xm
    row_y_ptr = y_ptr + row * stride_ym

    # ----- Pass 1: accumulate sum of squares in float32 -----
    sum_sq = tl.zeros((), dtype=tl.float32)

    col_start = 0
    while col_start < D:
        cols = col_start + offs_n
        col_mask = cols < D
        mask = row_mask & col_mask

        x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_sq += tl.sum(x_f32 * x_f32, axis=0)

        col_start += BLOCK_N

    # L2 norm (float32)
    norm = tl.sqrt(sum_sq)

    # ----- Pass 2: normalize -----
    col_start = 0
    while col_start < D:
        cols = col_start + offs_n
        col_mask = cols < D
        mask = row_mask & col_mask

        x = tl.load(row_x_ptr + cols * stride_xn, mask=mask, other=0.0)
        # Broadcast scalar `norm` across vector; Triton will handle type promotion.
        y = x / norm
        tl.store(row_y_ptr + cols * stride_yn, y, mask=mask)

        col_start += BLOCK_N


def triton_l2norm(x: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize along dim=1 using a fused Triton kernel.
    Expects 2D input [B, D] on CUDA.
    """
    assert x.ndim == 2, "Only 2D tensors are supported"
    B, D = x.shape
    if B == 0:
        return x.clone()

    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    stride_xm, stride_xn = x_contig.stride()
    stride_ym, stride_yn = y.stride()

    BLOCK_N = 256

    # One program per row
    grid = lambda meta: (max(1, B),)

    l2norm_fused_kernel[grid](
        x_contig,
        y,
        B,
        D,
        stride_xm,
        stride_xn,
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
