# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_maxpool_sum_scale_kernel(
    a_ptr, b_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    scale,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel:
      y = a @ b + bias
      y -> MaxPool1d(kernel_size=KERNEL_SIZE, stride=KERNEL_SIZE) over N
      out[m] = scale * sum_p max_window(y[m, p*KERNEL_SIZE:(p+1)*KERNEL_SIZE])

    a: (M, K)
    b: (K, N)  (transposed Linear weight)
    bias: (N,)
    out: (M,)
    """

    # ---------------------------------------------
    # Tiling indices
    # ---------------------------------------------
    pid_m = tl.program_id(0)  # tile id along M
    pid_n = tl.program_id(1)  # tile id along N

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    mask_m = offs_m < M
    mask_n = offs_n < N

    # ---------------------------------------------
    # Pointers for A and B tiles
    # ---------------------------------------------
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak  # [BM, BK]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn  # [BK, BN]

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---------------------------------------------
    # Blocked matmul along K
    # ---------------------------------------------
    for k in range(0, K, BLOCK_K):
        # Masks for the current K-tile
        k_mask = offs_k[None, :] < (K - k)  # [1, BK]

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask,  # [BM, BK]
            other=0.0,
        )

        # For B we want a [BK, BN] mask
        k_mask_b = offs_k[:, None] < (K - k)  # [BK, 1]
        b = tl.load(
            b_ptrs,
            mask=k_mask_b & mask_n[None, :],  # [BK, BN]
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---------------------------------------------
    # Add bias (broadcast over rows)
    # ---------------------------------------------
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)  # [BN]
    acc += bias[None, :]  # [BM, BN]

    # ---------------------------------------------
    # Fused MaxPool1d + sum + scale over N
    # ---------------------------------------------
    # Only complete windows are kept: same as MaxPool1d(kernel_size, stride=kernel_size, ceil_mode=False)
    valid_N = (N // KERNEL_SIZE) * KERNEL_SIZE

    tile_n_start = pid_n * BLOCK_N
    local_n = tl.arange(0, BLOCK_N)  # 0..BLOCK_N-1
    global_n = tile_n_start + local_n

    # Columns that belong to at least one valid pooling window
    valid_col_mask = (global_n < valid_N) & mask_n  # [BN]

    row_partial_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Number of pooling windows per tile (BLOCK_N guaranteed % KERNEL_SIZE == 0)
    NUM_WIN_TILE = BLOCK_N // KERNEL_SIZE

    # Loop over pooling windows inside this N-tile
    for w in tl.static_range(0, NUM_WIN_TILE):
        window_start = w * KERNEL_SIZE
        window_end = window_start + KERNEL_SIZE

        in_window = (local_n >= window_start) & (local_n < window_end)  # [BN]
        col_mask = in_window & valid_col_mask                            # [BN]

        # Select only elements in this window and within valid_N; others set to -inf
        window_vals = tl.where(col_mask[None, :], acc, -1e30)  # [BM, BN]

        # Max over N-dim within the window
        max_vals = tl.max(window_vals, axis=1)  # [BM]

        # This window is globally valid iff its starting index is < valid_N
        is_window_valid = (tile_n_start + window_start) < valid_N

        # Accumulate only valid windows; invalid ones contribute 0
        row_partial_sum += tl.where(is_window_valid, max_vals, 0.0)

    # ---------------------------------------------
    # Atomic add the scaled partial sums into output
    # ---------------------------------------------
    tl.atomic_add(
        out_ptr + offs_m,
        row_partial_sum * scale,
        mask=mask_m,
    )


def fused_linear_maxpool_sum_scale(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    kernel_size: int,
    scale_factor: float,
) -> torch.Tensor:
    """
    x: (M, K)
    weight: (N, K)  (same layout as nn.Linear.weight)
    bias: (N,)
    returns: (M,)  after Linear -> MaxPool1d(kernel_size, stride=kernel_size) -> sum -> scale
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]
    assert bias.shape[0] == N
    assert kernel_size > 0

    # Triton matmul expects B with shape (K, N)
    w_t = weight.contiguous().T  # (K, N)

    out = torch.zeros((M,), device=x.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Ensure pooling windows do not cross tile boundaries
    assert BLOCK_N % kernel_size == 0

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    linear_maxpool_sum_scale_kernel[grid](
        x, w_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        float(scale_factor),
        KERNEL_SIZE=kernel_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return out.to(x.dtype)


class ModelNew(nn.Module):
    """
    Triton-optimized version:
      x -> Linear(in_features, out_features)
        -> MaxPool1d(kernel_size, stride=kernel_size) over feature dim
        -> sum over feature dim
        -> scale_factor
    Returns: (batch_size,)
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        # Match nn.Linear parameter shapes
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.kernel_size = int(kernel_size)
        self.scale_factor = float(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_features)
        out = fused_linear_maxpool_sum_scale(
            x, self.weight, self.bias,
            self.kernel_size,
            self.scale_factor,
        )  # (batch_size,)
        return out
