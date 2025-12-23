import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Higher throughput on 4090 for large tensors, still very low register use
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        # Slightly smaller tile as a safe fallback when occupancy / registers differ
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
        # Extra config to exploit more warps when it helps memory latency hiding
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
    ],
    key=["numel"],
)
@triton.jit
def mul_leakyrelu_inplace_kernel(
    x_ptr,
    numel,
    multiplier,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    """
    In-place: x = leaky_relu(x * multiplier, negative_slope)
    x is assumed to be a contiguous 1D buffer of length numel.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # Cheap ops, fully fused; register footprint stays very small
    x = x * multiplier
    x = tl.where(x >= 0, x, x * negative_slope)
    tl.store(x_ptr + offs, x, mask=mask)


def mul_leakyrelu_inplace(x, multiplier, negative_slope):
    """
    Apply: x <- leaky_relu(x * multiplier, negative_slope) in-place using Triton.
    x: [M, N] contiguous tensor (output of F.linear).
    """
    assert x.is_cuda, "Input must be on CUDA"

    if not x.is_contiguous():
        x = x.contiguous()

    numel = x.numel()

    # Ensure grid size is always > 0 even for numel == 0
    grid = lambda META: (max(1, triton.cdiv(numel, META["BLOCK_SIZE"])),)

    mul_leakyrelu_inplace_kernel[grid](
        x,
        numel,
        float(multiplier),
        float(negative_slope),
    )
    return x


class ModelNew(nn.Module):
    """
    Optimized replacement for:
      Linear(in_features, out_features) + scalar multiply + LeakyReLU

    Strategy:
      - Use cuBLAS-backed GEMM via torch.nn.functional.linear for the Linear.
      - Fuse scalar multiply and LeakyReLU into a single Triton epilogue kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        # x: [batch_size, in_features]
        y = nn.functional.linear(x, self.weight, self.bias)
        y = mul_leakyrelu_inplace(y, self.multiplier, self.negative_slope)
        return y
