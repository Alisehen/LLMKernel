import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_linear_relu_div_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    scale,  # 1.0 / divisor
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first K-tile of X and W
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < (K - k)
        x_mask = (offs_m[:, None] < M) & k_mask
        w_mask = k_mask.T & (offs_n[None, :] < N)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w, allow_tf32=True)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Divide by constant (multiply by precomputed inverse scale)
    acc = acc * scale

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=mask_out)


def fused_linear_relu_div(x, weight, bias, divisor):
    """
    x:      [M, K]
    weight: [N, K] (PyTorch Linear weight layout is [N, K] = [out_features, in_features])
    bias:   [N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"

    M, K = x.shape
    N = weight.shape[0]
    # PyTorch Linear: y = x @ weight.T + bias
    # Use W_t for efficient row-major matmul: X[M,K] @ W_t[K,N]
    w_t = weight.t().contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    scale = 1.0 / float(divisor)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_linear_relu_div_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        scale,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=8,
        num_stages=3,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of:
        y = ReLU(x @ W.T + b) / divisor
    """
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.divisor = divisor

    def forward(self, x):
        return fused_linear_relu_div(x, self.weight, self.bias, self.divisor)
