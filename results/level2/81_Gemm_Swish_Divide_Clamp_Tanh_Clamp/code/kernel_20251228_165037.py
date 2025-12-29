import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def fused_linear_swish_div_clamp_tanh_clamp_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for A and B tiles
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # GEMM loop
    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k

        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Optional bias add (per output feature)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    # ---- Fused elementwise pipeline ----
    # Swish: x * sigmoid(x)  with sigmoid(x) = 1 / (1 + exp(-x))
    neg_acc = -acc
    exp_neg = tl.exp(neg_acc)
    sig = 1.0 / (1.0 + exp_neg)
    acc = acc * sig

    # Divide by 2.0
    acc = acc * 0.5

    # Clamp between -1 and 1
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # Tanh via definition: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    exp_2x = tl.exp(acc * 2.0)
    acc = (exp_2x - 1.0) / (exp_2x + 1.0)

    # Final clamp between -1 and 1
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


def fused_linear_swish_div_clamp_tanh_clamp(x: torch.Tensor,
                                            weight: torch.Tensor,
                                            bias: torch.Tensor | None):
    """
    x:      [M, K]
    weight: [N, K]  (same as nn.Linear.weight)
    bias:   [N] or None
    returns: [M, N]
    """
    # CPU fallback to maintain correctness in non-CUDA environments
    if not x.is_cuda:
        y = nn.functional.linear(x, weight, bias)
        y = y * torch.sigmoid(y)
        y = y / 2.0
        y = torch.clamp(y, min=-1.0, max=1.0)
        y = torch.tanh(y)
        y = torch.clamp(y, min=-1.0, max=1.0)
        return y

    M, K = x.shape
    N = weight.shape[0]

    # Prepare output
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Convert weight to [K, N] for GEMM
    b = weight.t().contiguous()

    HAS_BIAS = bias is not None

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    fused_linear_swish_div_clamp_tanh_clamp_kernel[grid](
        x, b, bias if HAS_BIAS else c, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        HAS_BIAS=HAS_BIAS,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    )
    return c


class ModelNew(nn.Module):
    """
    Triton-optimized version of the reference model:
    GEMM (linear) + swish + divide + clamp + tanh + clamp.
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
