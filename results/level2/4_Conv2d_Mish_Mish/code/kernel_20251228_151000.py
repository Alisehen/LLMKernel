# Optimized Triton code for fused Mish(Mish(x)) on RTX 4090

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def mish2_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    In-place fused Mish(Mish(x)) kernel.
    1D grid over the flat tensor; all fused ops share the same offsets & mask.
    Uses a numerically-stable, branch-free softplus and tanh.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load original values (contiguous 1D view)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_fp = x.to(tl.float32)

    # ---------- Mish #1: y = x * tanh(softplus(x)) ----------
    # softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
    # This is numerically stable and requires only 1 exp + 1 log.
    abs_x = tl.abs(x_fp)
    sp1 = tl.log(1.0 + tl.exp(-abs_x)) + tl.maximum(x_fp, 0.0)

    # tanh(sp1) for sp1 >= 0:
    # tanh(z) = 1 - 2 / (exp(2z) + 1), stable for large z and 1 exp + 1 div.
    t1 = tl.exp(2.0 * sp1)
    tanh_sp1 = 1.0 - 2.0 / (t1 + 1.0)

    y_fp = x_fp * tanh_sp1

    # ---------- Mish #2: z = y * tanh(softplus(y)) ----------
    abs_y = tl.abs(y_fp)
    sp2 = tl.log(1.0 + tl.exp(-abs_y)) + tl.maximum(y_fp, 0.0)

    t2 = tl.exp(2.0 * sp2)
    tanh_sp2 = 1.0 - 2.0 / (t2 + 1.0)

    z_fp = y_fp * tanh_sp2

    # Cast back to original dtype and store in-place
    z = z_fp.to(x.dtype)
    tl.store(x_ptr + offsets, z, mask=mask)


def mish_mish_inplace(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Mish(Mish(x)) to `x` in-place using an optimized Triton kernel.
    Assumes `x` is on CUDA and contiguous.
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    assert x.is_contiguous(), "Activation kernel assumes contiguous tensor"

    n_elements = x.numel()
    if n_elements == 0:
        return x

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    mish2_kernel[grid](
        x,
        n_elements,
    )
    return x


def conv2d_mish2(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Hybrid implementation:
    - Convolution via cuDNN (torch.nn.functional.conv2d)
    - Fused Mish(Mish(.)) in a single Triton kernel pass
    """
    y = torch.nn.functional.conv2d(
        x,
        weight,
        bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    mish_mish_inplace(y)
    return y


class ModelNew(nn.Module):
    """
    Optimized model:
    - Convolution via cuDNN (torch.nn.functional.conv2d)
    - Fused Mish + Mish via a Triton pointwise kernel
    Behavior matches:
        y = conv2d(x)
        y = mish(y)
        y = mish(y)
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        assert isinstance(kernel_size, int), "Only integer kernel_size is supported"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Match the parameter interface of the previous custom conv:
        # weight: [Cout, Cin, Kh, Kw], bias: [Cout]
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d_mish2(x, self.weight, self.bias)
