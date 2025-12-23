import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def fused_linear_swish_scale_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale,  # scalar float
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program instance
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Create pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Mask to guard out-of-bounds K
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)  # [BLOCK_N]
    acc += bias[None, :]

    # Swish activation: x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-acc))
    acc = acc * sig

    # Scale
    acc = acc * scale

    # Write back
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_out)


def fused_linear_swish_scale(x, weight, bias, scale: float):
    """
    x: [M, K]
    weight: [N, K] (nn.Linear.weight, i.e., out_features x in_features)
    bias: [N]
    scale: scalar float
    Returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    M, K = x.shape
    N = weight.shape[0]

    # Prepare weight as [K, N] row-major for efficient GEMM with A[M, K]
    b = weight.t().contiguous()  # [K, N]

    c = torch.empty((M, N), device=x.device, dtype=torch.float32)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    fused_linear_swish_scale_kernel[grid](
        x, b, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        scale,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=8,
        num_stages=4,
    )
    return c


class ModelNew(nn.Module):
    """
    Triton-optimized version of the model:
    y = (x @ W^T + b) * sigmoid(x @ W^T + b) * scaling_factor
    with matmul + bias + Swish + scaling fused in a single Triton kernel.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        # Use nn.Linear only to manage parameters / initialization
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        return fused_linear_swish_scale(
            x,
            self.linear.weight,
            self.linear.bias,
            self.scaling_factor,
        )
