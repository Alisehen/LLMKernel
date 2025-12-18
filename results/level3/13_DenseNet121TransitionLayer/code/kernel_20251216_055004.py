import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_gemm_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer to the first K-tile of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k_iter = 0
    while k_iter < K:
        k_mask = offs_k[None, :] + k_iter < K

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & k_mask.T,
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_iter += BLOCK_K

    # Write back result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def conv1x1_triton(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Efficient 1x1 convolution implemented as a batched GEMM in Triton.

    x:      [B, C_in, H, W]
    weight: [C_out, C_in, 1, 1] (same layout as nn.Conv2d with kernel_size=1, bias=False)
    """
    assert x.ndim == 4
    assert weight.ndim == 4
    B, C_in, H, W = x.shape
    C_out, C_in_w, kH, kW = weight.shape
    assert C_in_w == C_in and kH == 1 and kW == 1, "Only 1x1 conv is supported"

    x = x.contiguous()
    weight = weight.contiguous()

    # Flatten spatial + batch: [B, C_in, H, W] -> [M, K] with M = B*H*W, K = C_in
    x_2d = x.permute(0, 2, 3, 1).reshape(-1, C_in).contiguous()
    M, K = x_2d.shape

    # Weight: [C_out, C_in, 1, 1] -> [C_in, C_out]
    w_2d = weight.view(C_out, C_in).contiguous()
    b_mat = w_2d.t().contiguous()  # [K, N] where N=C_out

    N = C_out
    out_2d = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
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
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        # Keep Conv2d module for parameter management (weight layout, init, state_dict)
        self.conv = nn.Conv2d(
            num_input_features,
            num_output_features,
            kernel_size=1,
            bias=False,
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        x = self.bn(x)
        x = self.relu(x)
        # Use Triton kernel for the 1x1 convolution
        x = conv1x1_triton(x, self.conv.weight)
        x = self.pool(x)
        return x
