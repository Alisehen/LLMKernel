# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Conservative baseline – good occupancy, low register pressure
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=2,
        ),
        # Latency-hiding, higher arithmetic intensity on Ada
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_warps=8,
            num_stages=3,
        ),
        # Skewed tile – favors large N (C_out) with good TC utilization
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_gemm_kernel(
    a_ptr,  # A: [M, K]
    b_ptr,  # B: [K, N]
    c_ptr,  # C: [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch grid
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers to first K-slice for this tile
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator kept in registers, fp32 for stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Help compiler with vectorization / alignment assumptions
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    k = 0
    while k < K:
        k_mask = (offs_k + k) < K

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0,
        )

        # Tensor Core-friendly matmul on Ada (TF32 for fp32 inputs)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Write back
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=out_mask)


def conv1x1_triton(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    High-performance 1x1 convolution implemented as GEMM using Triton.

    x:      [B, C_in, H, W]          (fp32)
    weight: [C_out, C_in, 1, 1]      (fp32, bias=False)
    """
    assert x.ndim == 4
    assert weight.ndim == 4
    B, C_in, H, W = x.shape
    C_out, C_in_w, kH, kW = weight.shape
    assert C_in_w == C_in and kH == 1 and kW == 1, "Only 1x1 conv is supported"

    # Ensure predictable layout
    x = x.contiguous()

    # Logical A matrix: [M, K] = [B*H*W, C_in]
    # Make channels the fastest-moving dimension to maximize coalescing.
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C_in]
    x_2d = x_nhwc.view(-1, C_in)                 # [M, K]
    M, K = x_2d.shape

    # Logical B matrix: [K, N] = [C_in, C_out]
    w_2d = weight.view(C_out, C_in)  # [C_out, C_in]
    b_mat = w_2d.t()                 # [K, N] (view, no copy)
    N = C_out

    # Output [M, N]
    out_2d = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    conv1x1_gemm_kernel[grid](
        x_2d, b_mat, out_2d,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        b_mat.stride(0), b_mat.stride(1),
        out_2d.stride(0), out_2d.stride(1),
    )

    # Reshape back to [B, C_out, H, W]
    out = out_2d.view(B, H, W, C_out).permute(0, 3, 1, 2).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            num_input_features,
            num_output_features,
            kernel_size=1,
            bias=False,
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = conv1x1_triton(x, self.conv.weight)
        x = self.pool(x)
        return x
