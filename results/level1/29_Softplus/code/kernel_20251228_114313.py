import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def softplus_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Piecewise softplus approximation
    # For large positive x: softplus(x) ≈ x
    # For large negative x: softplus(x) ≈ exp(x)
    # For moderate x: numerically stable formulation:
    #   softplus(x) = max(x, 0) + log(1 + exp(-|x|))
    T_POS = 20.0
    T_NEG = -20.0

    pos_mask_val = x > T_POS
    neg_mask_val = x < T_NEG
    mid_mask_val = ~(pos_mask_val | neg_mask_val)

    abs_x = tl.abs(x)
    s = tl.maximum(x, 0.0)

    # One exp for all elements; reused for both mid and neg regions.
    exp_term = tl.exp(-abs_x)

    # Mid-region: stable softplus
    soft_mid = s + tl.log(1.0 + exp_term)

    # Negative-region approximation: softplus(x) ≈ exp(x)
    # For x < 0, exp(-abs_x) == exp(x), so we can reuse exp_term.
    soft_neg = exp_term

    # Combine piecewise results
    out = tl.where(
        pos_mask_val,
        x,
        tl.where(neg_mask_val, soft_neg, soft_mid),
    )

    tl.store(y_ptr + offsets, out, mask=mask)


def triton_softplus(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 1024  # power-of-2
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    softplus_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return y


class ModelNew(nn.Module):
    """
    Simple model that performs a Softplus activation using a high-performance Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_softplus(x)
