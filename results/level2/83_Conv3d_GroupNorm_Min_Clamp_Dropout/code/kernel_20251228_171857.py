import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fill_min_value_kernel(
    y_ptr,
    numel,
    min_value,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel that fills the output tensor with a constant min_value.

    This exploits the identity:
        y = torch.min(x, min_value)
        y = torch.clamp(y, min=min_value, max=max_value)
    =>  y is identically min_value for all elements, regardless of x.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Broadcast scalar min_value to all valid positions
    tl.store(y_ptr + offs, min_value, mask=mask)


def groupnorm_min_clamp(x, weight, bias, groups, min_value, max_value, eps=1e-5):
    """
    Optimized fused GroupNorm + torch.min(x, min_value) + torch.clamp(x, min=min_value, max=max_value)
    for 5D tensors in NCDHW layout.

    For the specific composition:
        x = GroupNorm(x)
        x = torch.min(x, min_value)
        x = torch.clamp(x, min=min_value, max=max_value)

    every element after min+clamp is exactly min_value, independent of the GroupNorm output.
    Thus we bypass all GroupNorm work and directly fill the output with min_value.
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."
    x = x.contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    numel = y.numel()
    BLOCK_SIZE = 256

    # Launch configuration: 1D grid over all elements
    grid = lambda META: (max(1, triton.cdiv(numel, META["BLOCK_SIZE"])),)

    fill_min_value_kernel[grid](
        y,
        numel,
        float(min_value),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, then (conceptually) Group Normalization,
    torch.min(x, min_value), torch.clamp(x, min=min_value, max=max_value),
    and finally dropout.

    The GroupNorm + min + clamp sequence is algebraically simplified:
    for any input, the result after these ops is identically min_value, so the
    Triton kernel directly writes min_value and skips all GroupNorm computation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.groups = groups
        # GroupNorm affine parameters (kept for API compatibility, but do not affect output)
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias = nn.Parameter(torch.zeros(out_channels))
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.eps = 1e-5
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv(x)
        # Fused and algebraically simplified GroupNorm + min + clamp
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
