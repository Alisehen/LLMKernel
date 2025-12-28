import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    T_POS: tl.constexpr,
    T_NEG: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Region masks (only on valid lanes)
    pos_mask = (x > T_POS) & mask        # large positive: softplus(x) ≈ x
    neg_mask = (x < T_NEG) & mask        # large negative: softplus(x) ≈ exp(x)
    mid_mask = (~pos_mask) & (~neg_mask) & mask  # remaining: exact, numerically-stable

    # Initialize result
    y = tl.zeros_like(x)

    # Large positive region: softplus(x) ≈ x (no exp/log)
    y = tl.where(pos_mask, x, y)

    # Large negative region: softplus(x) ≈ exp(x)
    # Avoid overflow by zeroing non-neg_mask lanes before exp
    x_neg = tl.where(neg_mask, x, 0.0)
    exp_neg = tl.exp(x_neg)
    y = tl.where(neg_mask, exp_neg, y)

    # Middle region: numerically-stable exact softplus
    # softplus(x) = max(x, 0) + log(1 + exp(-abs(x)))
    x_mid = tl.where(mid_mask, x, 0.0)
    abs_x = tl.abs(x_mid)
    z = tl.exp(-abs_x)
    soft_mid = tl.maximum(x_mid, 0.0) + tl.log(1.0 + z)
    y = tl.where(mid_mask, soft_mid, y)

    # Store result
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_softplus(x: torch.Tensor) -> torch.Tensor:
    # Ensure contiguous
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n_elements = x_contig.numel()
    BLOCK_SIZE = 256

    # Grid size must be > 0 even when n_elements == 0
    grid = lambda meta: (triton.cdiv(max(n_elements, 1), meta["BLOCK_SIZE"]),)

    # Thresholds chosen similar to PyTorch Softplus defaults (beta=1, threshold=20)
    T_POS = 20.0
    T_NEG = -20.0

    softplus_kernel[grid](
        x_contig,
        y,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        T_POS=T_POS,
        T_NEG=T_NEG,
    )

    return y.view_as(x)


class ModelNew(nn.Module):
    """
    Simple model that performs a Softplus activation using a high-performance Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softplus(x)
