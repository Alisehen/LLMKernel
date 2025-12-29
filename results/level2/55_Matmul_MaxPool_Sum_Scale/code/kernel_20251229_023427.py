import torch, torch.nn as nn, triton, triton.language as tl


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
    # Program ids for tiling along M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < (K - k)
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_n[None, :] & k_mask.T,
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # --- Fused MaxPool1d + sum + scale over N dimension ---

    # We only pool over complete windows: valid_N = (N // KERNEL_SIZE) * KERNEL_SIZE
    valid_N = (N // KERNEL_SIZE) * KERNEL_SIZE
    tile_n_start = pid_n * BLOCK_N

    # Per-row partial sum of pooled values from this N-tile
    row_partial_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # Current running max within a pooling window for each row
    cur_max = tl.full((BLOCK_M,), -1e30, dtype=tl.float32)

    # Iterate over local N within the tile; BLOCK_N is a compile-time constant
    for ln in range(0, BLOCK_N):
        global_n = tile_n_start + ln

        # Is this column part of any valid pooling window?
        is_valid_col = global_n < valid_N

        # Load the column of the GEMM result from registers
        vals = acc[:, ln]  # [BLOCK_M]

        # Update running max only for valid columns
        cur_max = tl.where(is_valid_col, tl.maximum(cur_max, vals), cur_max)

        # Is this the last element in a pooling window?
        is_last = is_valid_col & (((global_n + 1) % KERNEL_SIZE) == 0)

        # If this column closes a window, accumulate its max into the row sum
        row_partial_sum += tl.where(is_last, cur_max, 0.0)

        # Reset running max at the end of each window
        cur_max = tl.where(is_last, -1e30, cur_max)

    # Atomic add the scaled partial sums into the output vector
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

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
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

    # Cast back to input dtype for consistency with PyTorch (Linear outputs x.dtype, then pooling/sum/scale)
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
