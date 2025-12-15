import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_gap_bias_logsumexp_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel: GlobalAvgPool + BiasAdd + LogSumExp + Sum + Multiply
    Input: [N, C, H, W] -> Output: [N, 1]
    """
    pid_n = tl.program_id(0)  # batch index

    if pid_n >= N:
        return

    HW = H * W
    offs_c = tl.arange(0, BLOCK_C)

    # Step 1: Global Average Pooling - compute mean over H, W for each channel
    gap_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for h in range(H):
        for w in range(W):
            x_ptrs = x_ptr + pid_n * stride_xn + offs_c * stride_xc + h * stride_xh + w * stride_xw
            x_vals = tl.load(x_ptrs, mask=offs_c < C, other=0.0)
            gap_acc += x_vals

    gap_acc = gap_acc / HW  # [C] - global average pooled values

    # Step 2: Add bias
    bias_vals = tl.load(bias_ptr + offs_c, mask=offs_c < C, other=0.0)
    gap_acc = gap_acc + bias_vals  # [C]

    # Step 3: LogSumExp over channels
    max_val = tl.max(gap_acc, axis=0)
    exp_vals = tl.exp(gap_acc - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    logsumexp_val = max_val + tl.log(sum_exp)

    # Step 4: Multiply by 10.0
    result = logsumexp_val * 10.0

    # Store result [N, 1]
    tl.store(out_ptr + pid_n, result)


def fused_post_convtranspose(x, bias):
    """
    Fused: GAP + Bias + LogSumExp + Sum + Multiply
    """
    N, C, H, W = x.shape
    out = torch.empty((N, 1), device=x.device, dtype=x.dtype)

    # BLOCK_C must be power of 2 and >= C
    BLOCK_C = triton.next_power_of_2(C)

    grid = (N,)
    fused_gap_bias_logsumexp_kernel[grid](
        x, bias, out,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_C=BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose2d (PyTorch native) + Fused post-ops (Triton)

    NOTE: ConvTranspose2d has complex index mapping - keep it in PyTorch.
    Only fuse the simpler subsequent operations in Triton.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose as PyTorch native - DO NOT reimplement in Triton
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(out_channels))  # Flatten for Triton

    def forward(self, x):
        # Step 1: PyTorch native ConvTranspose2d
        x = self.conv_transpose(x)
        # Step 2: Fused post-ops in Triton
        x = fused_post_convtranspose(x, self.bias)
        return x
