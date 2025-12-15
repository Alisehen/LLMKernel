import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel:
      1. Softmax over channel dimension (dim=1) per (n, h, w)
      2. Add channel bias (C, 1, 1)
      3. Multiply by scaling_factor
      4. Apply sigmoid

    Input / Output: [N, C, H, W]
    Softmax is computed over C for each fixed (n, h, w).
    """
    pid = tl.program_id(0)
    num_positions = N * H * W

    if pid >= num_positions:
        return

    # Decode linear pid into (n, h, w)
    HW = H * W
    n = pid // HW
    hw = pid % HW
    h = hw // W
    w = hw % W

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base_offset = n * stride_n + h * stride_h + w * stride_w
    x_ptrs = x_ptr + base_offset + offs_c * stride_c

    # Load input channels
    x_vals = tl.load(x_ptrs, mask=mask_c, other=0.0)

    # Compute softmax over channels (numerically stable)
    # Mask out invalid channels with -inf so they don't affect max/sum
    x_masked = tl.where(mask_c, x_vals, -float("inf"))
    x_fp32 = tl.cast(x_masked, tl.float32)

    max_val = tl.max(x_fp32, axis=0)
    x_shifted = x_fp32 - max_val
    exp_x = tl.exp(x_shifted)
    exp_x = tl.where(mask_c, exp_x, 0.0)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax = exp_x / sum_exp  # [BLOCK_C]

    # Add bias (C,1,1) flattened to (C,)
    bias_vals = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    bias_fp32 = tl.cast(bias_vals, tl.float32)

    y = softmax + bias_fp32
    y = y * scaling_factor

    # Sigmoid: 1 / (1 + exp(-y))
    neg_y = -y
    exp_neg_y = tl.exp(neg_y)
    denom = 1.0 + exp_neg_y
    sigmoid = 1.0 / denom

    # Cast back to original dtype
    out_vals = tl.cast(sigmoid, x_vals.dtype)
    tl.store(out_ptr + base_offset + offs_c * stride_c, out_vals, mask=mask_c)


def fused_post_convtranspose(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    """
    x: [N, C, H, W] - result of ConvTranspose2d
    bias: [C, 1, 1] or [C] - channel-wise bias
    Returns: [N, C, H, W] with fused operations:
        softmax(dim=1) -> +bias -> *scaling_factor -> sigmoid
    """
    assert x.is_cuda, "Input must be on CUDA device"
    N, C, H, W = x.shape

    # Ensure bias is [C]
    bias_flat = bias.view(C)

    out = torch.empty_like(x)

    # BLOCK_C must be power of 2 and >= C (for tl.arange)
    # Use next power of 2, capped to a reasonable size for stability
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    # Optional cap to avoid very large blocks; for typical CNN C this won't trigger
    if BLOCK_C > 1024:
        BLOCK_C = 1024

    grid = (N * H * W,)

    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias_flat, out,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    """
    ConvTranspose2d (PyTorch native) +
    fused softmax(dim=1) + bias add + scaling + sigmoid (Triton).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose2d in PyTorch as required
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        # Match original bias shape (C, 1, 1)
        self.bias = nn.Parameter(torch.randn(out_channels, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post_convtranspose(x, self.bias, self.scaling_factor)
        return x
