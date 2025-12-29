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
    # Program ids for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in FP32 for numerical accuracy
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        k_cur = k + offs_k  # [BLOCK_K]

        # Pointers for X: [M, K] with strides (stride_xm, stride_xk)
        x_ptrs = x_ptr + (
            offs_m[:, None] * stride_xm + k_cur[None, :] * stride_xk
        )
        # Pointers for W: [K, N] with strides (stride_wk, stride_wn)
        w_ptrs = w_ptr + (
            k_cur[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )

        # Masks
        x_mask = (offs_m[:, None] < M) & (k_cur[None, :] < K)
        w_mask = (k_cur[:, None] < K) & (offs_n[None, :] < N)

        # Loads
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # FMA
        acc += tl.dot(x, w, allow_tf32=False)

    # Add bias (broadcast over rows)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    bias = tl.astype(bias, tl.float32)
    acc += bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Divide by constant (multiply by precomputed inverse scale)
    acc = acc * scale

    # Write back
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=mask_out)


def fused_linear_relu_div(x, weight, bias, divisor):
    """
    Fused implementation of:
        y = ReLU(x @ W + b) / divisor

    x:      [M, K]
    weight: [K, N]  (GEMM layout: [in_features, out_features])
    bias:   [N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"

    M, K = x.shape
    K_w, N = weight.shape
    assert K_w == K, "Incompatible shapes: x is [M, K], weight must be [K, N]"
    assert bias.shape[0] == N, "Bias must have shape [N]"

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    scale = 1.0 / float(divisor)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_linear_relu_div_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
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
        y = ReLU(x @ linear.weight.T + linear.bias) / divisor

    PyTorch reference:
        linear = nn.Linear(in_features, out_features)
        y = torch.relu(x @ linear.weight.T + linear.bias) / divisor
    """

    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        # Match reference module structure for easy state_dict loading
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        # x: [batch_size, in_features] = [M, K]
        # linear.weight: [out_features, in_features] = [N, K]
        # We need weight in [K, N] for the Triton kernel
        weight_t = self.linear.weight.t().contiguous()  # [K, N]
        bias = self.linear.bias                         # [N]
        return fused_linear_relu_div(x, weight_t, bias, self.divisor)
