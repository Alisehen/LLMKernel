import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline, conservative: good occupancy, low register pressure
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # More compute per tile, still moderate register usage
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Asymmetric tile for wide-N cases, slightly deeper pipeline
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_div_gelu_kernel(
    a_ptr, w_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cm, stride_cn,
    inv_divisor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks
    m_mask = offs_m < M
    n_mask = offs_n < N
    mn_mask = m_mask[:, None] & n_mask[None, :]

    # Pointers for A: [M, K]
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # Pointers for W: [N, K], accessed as B[K, N] with B(k, n) = W(n, k)
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K

        a = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        # Tensor-core friendly dot
        acc += tl.dot(a, w, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # Bias add: bias shape [N], broadcast along M
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # Divide by scalar via multiplication by inverse
    acc *= inv_divisor

    # GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))
    t = acc * 0.7071067811865476  # 1 / sqrt(2)
    t = tl.math.erf(t)
    acc = 0.5 * acc * (1.0 + t)

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mn_mask)


def fused_linear_div_gelu(x: torch.Tensor,
                          weight: torch.Tensor,
                          bias: torch.Tensor,
                          divisor: float) -> torch.Tensor:
    """
    Fused implementation of:
        y = x @ weight.T + bias
        y = y / divisor
        y = GELU(y)

    Shapes:
        x:      [M, K]
        weight: [N, K]  (same as nn.Linear.weight)
        bias:   [N]
        out:    [M, N]
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert weight.is_cuda and bias.is_cuda, "Parameters must be on CUDA device"
    assert x.dtype == weight.dtype == bias.dtype, "dtypes of x, weight, bias must match"

    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, "weight shape must be [N, K]"
    assert bias.shape[0] == N, "bias shape must match output features"

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    inv_divisor = 1.0 / float(divisor)

    linear_div_gelu_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        # Map W [N, K] as B [K, N]:
        # B(k, n) = W(n, k) -> stride_wk = stride over K, stride_wn = stride over N
        weight.stride(1), weight.stride(0),
        y.stride(0), y.stride(1),
        inv_divisor,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of:
        y = Linear(x)
        y = y / divisor
        y = GELU(y)
    """

    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.divisor = float(divisor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
            self.weight.data = self.weight.data.cuda()
            self.bias.data = self.bias.data.cuda()
        return fused_linear_div_gelu(x, self.weight, self.bias, self.divisor)
