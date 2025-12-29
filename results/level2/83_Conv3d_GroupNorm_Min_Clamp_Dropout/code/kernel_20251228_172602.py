# corrected code

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["numel"],
)
@triton.jit
def fill_min_value_kernel(
    y_ptr,              # *T
    numel,              # int32
    min_value,          # scalar (same dtype as *y_ptr)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that fills the output tensor with a constant `min_value`.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Single, coalesced, masked store of the constant value
    tl.store(y_ptr + offs, min_value, mask=mask)


def groupnorm_min_clamp(x, weight, bias, groups, min_value, max_value, eps=1e-5):
    """
    Fused GroupNorm + torch.min + torch.clamp specialized for the identity:

        y = GroupNorm(x, weight, bias, groups, eps)
        y = torch.min(y, min_value)
        y = torch.clamp(y, min=min_value, max=max_value)

    For all inputs, the result of min+clamp is identically `min_value`,
    independent of GroupNorm output. We therefore bypass all computation and
    directly fill the output tensor with `min_value` using an optimized Triton
    kernel.
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."

    # Ensure a predictable, contiguous layout for linear indexing
    x = x.contiguous()

    # Allocate output tensor with same shape and dtype as x
    y = torch.empty_like(x)

    numel = y.numel()

    # 1D grid over all elements in the output tensor; ensure grid > 0
    def grid(meta):
        return (max(1, triton.cdiv(numel, meta["BLOCK_SIZE"])),)

    # Normalize min_value to a Python scalar so Triton sees a scalar, not a tensor
    if isinstance(min_value, torch.Tensor):
        min_scalar = float(min_value.item())
    else:
        min_scalar = float(min_value)

    # Launch kernel; autotuner selects best BLOCK_SIZE / num_warps
    fill_min_value_kernel[grid](
        y,            # y_ptr
        numel,        # numel
        min_scalar,   # scalar min_value (cast inside kernel to y.dtype)
    )
    return y


class ModelNew(nn.Module):
    """
    Model:
        x -> Conv3d -> (conceptual) GroupNorm -> min -> clamp -> Dropout

    The GroupNorm + min + clamp sequence is algebraically simplified:
      - For any input, the output of min+clamp is exactly `min_value`.
      - We implement this as a single Triton kernel that fills the tensor
        with `min_value`, achieving maximal throughput.

    Dropout is then applied using PyTorch's implementation.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        min_value,
        max_value,
        dropout_p,
    ):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.groups = groups

        # Kept for API compatibility; they do not affect the optimized path
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))

        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.eps = 1e-5
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv(x)

        # Fused & algebraically simplified GroupNorm + min + clamp
        x = groupnorm_min_clamp(
            x,
            self.gn_weight,
            self.gn_bias,
            self.groups,
            self.min_value,
            self.max_value,
            self.eps,
        )

        # Dropout applied after the constant activation
        x = self.dropout(x)
        return x
