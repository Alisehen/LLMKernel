import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_mish2_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C_in, H, W,
    C_out, KH, KW,
    H_out, W_out,
    K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = B * H_out * W_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C_out

    # Map linear output index -> (b, oh, ow)
    hw_out = H_out * W_out
    n_idx = offs_m // hw_out
    rem_m = offs_m % hw_out
    oh = rem_m // W_out
    ow = rem_m % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Map K index -> (ci, kh, kw)
        kk_hw = KH * KW
        ci = offs_k // kk_hw
        rem_k = offs_k % kk_hw
        kh = rem_k // KW
        kw = rem_k % KW

        # Input pointers (B, C_in, H, W)
        x_ptrs = (
            x_ptr
            + n_idx[:, None] * stride_xn
            + ci[None, :] * stride_xc
            + (oh[:, None] + kh[None, :]) * stride_xh
            + (ow[:, None] + kw[None, :]) * stride_xw
        )

        # Weight pointers (C_out, C_in, KH, KW)
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wo
            + ci[:, None] * stride_wc
            + kh[:, None] * stride_wkh
            + kw[:, None] * stride_wkw
        )

        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(x, w, allow_tf32=True)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bias[None, :]

    # First Mish: x * tanh(softplus(x)), softplus(x) = log(1 + exp(x))
    exp_acc = tl.exp(acc)
    sp = tl.log(1.0 + exp_acc)
    t2 = tl.exp(2.0 * sp)
    th = (t2 - 1.0) / (t2 + 1.0)
    acc = acc * th

    # Second Mish on result
    exp_acc2 = tl.exp(acc)
    sp2 = tl.log(1.0 + exp_acc2)
    t22 = tl.exp(2.0 * sp2)
    th2 = (t22 - 1.0) / (t22 + 1.0)
    acc = acc * th2

    # Store result
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + oh[:, None] * stride_yh
        + ow[:, None] * stride_yw
    )

    tl.store(
        y_ptrs,
        acc.to(tl.float32),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def conv2d_mish2(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    # Assumes NCHW, groups=1, stride=1, padding=0
    B, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w, "Input channels must match weight channels"

    H_out = H - KH + 1
    W_out = W - KW + 1

    x_ = x.contiguous()
    w_ = weight.contiguous()
    b_ = bias.contiguous()

    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=torch.float32)

    K = C_in * KH * KW

    def grid(META):
        return (
            triton.cdiv(B * H_out * W_out, META["BLOCK_M"]),
            triton.cdiv(C_out, META["BLOCK_N"]),
        )

    conv2d_mish2_kernel[grid](
        x_, w_, b_, y,
        B, C_in, H, W,
        C_out, KH, KW,
        H_out, W_out,
        K,
        x_.stride(0), x_.stride(1), x_.stride(2), x_.stride(3),
        w_.stride(0), w_.stride(1), w_.stride(2), w_.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version:
    Conv2d (stride=1, padding=0, groups=1) + Mish + Mish fused in a single kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        return conv2d_mish2(x, self.weight, self.bias)
