import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def epilogue_swish_div_clamp_tanh_clamp_kernel(
    x_ptr,  # [M, N] input (linear output), will be modified in-place
    M, N,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused elementwise epilogue:
      x = x * sigmoid(x)
      x = x / 2
      x = clamp(x, -1, 1)
      x = tanh(x)
      x = clamp(x, -1, 1)
    Applied over a 2D tensor [M, N] with given strides.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Compute pointers for this tile
    x_tile_ptr = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Load
    x = tl.load(x_tile_ptr, mask=mask, other=0.0)

    # Swish: x * sigmoid(x)
    x = x * tl.sigmoid(x)

    # Divide by 2
    x = x * 0.5

    # Clamp to [-1, 1]
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)

    # Tanh
    x = tl.tanh(x)

    # Final clamp to [-1, 1]
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)

    # Store back (in-place)
    tl.store(x_tile_ptr, x, mask=mask)


def fused_linear_swish_div_clamp_tanh_clamp(x: torch.Tensor,
                                            weight: torch.Tensor,
                                            bias: torch.Tensor | None):
    """
    x:      [M, K]
    weight: [N, K]  (same as nn.Linear.weight)
    bias:   [N] or None
    returns: [M, N]

    Implementation:
      1) Use PyTorch's highly-optimized linear (cuBLAS/cuBLASLt) for GEMM.
      2) Apply all elementwise ops in a single Triton kernel (epilogue).
    """
    # CPU fallback for correctness on non-CUDA tensors
    if not x.is_cuda:
        y = nn.functional.linear(x, weight, bias)
        y = y * torch.sigmoid(y)          # Swish
        y = y / 2.0
        y = torch.clamp(y, min=-1.0, max=1.0)
        y = torch.tanh(y)
        y = torch.clamp(y, min=-1.0, max=1.0)
        return y

    # 1) GEMM via vendor-optimized library
    y = nn.functional.linear(x, weight, bias)  # [M, N], contiguous row-major

    M, N = y.shape
    stride_m, stride_n = y.stride()

    # 2) Fused epilogue in Triton (in-place on y)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    epilogue_swish_div_clamp_tanh_clamp_kernel[grid](
        y,
        M, N,
        stride_m, stride_n,
        BLOCK_M=64,
        BLOCK_N=128,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of the reference model:
      y = Linear(x)
      y = swish(y)
      y = y / 2
      y = clamp(y, -1, 1)
      y = tanh(y)
      y = clamp(y, -1, 1)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_linear_swish_div_clamp_tanh_clamp(x, self.weight, self.bias)
