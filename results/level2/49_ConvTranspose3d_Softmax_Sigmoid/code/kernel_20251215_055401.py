import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def softmax_sigmoid_5d_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
):
    """
    Fused Softmax (dim=1 over channels) + Sigmoid for 5D tensor [N, C, D, H, W].

    For each (n, d, h, w), computes:
        y[n, c, d, h, w] = sigmoid(softmax(x[n, :, d, h, w])[c])
    """
    pid = tl.program_id(0)

    # Decode linear pid -> (n, d, h, w)
    spatial = D * H * W
    hw = H * W

    n = pid // spatial
    tmp = pid - n * spatial
    d = tmp // hw
    tmp = tmp - d * hw
    h = tmp // W
    w = tmp - h * W

    # Base pointer offset for (n, :, d, h, w)
    base_offset = (
        n * stride_n +
        d * stride_d +
        h * stride_h +
        w * stride_w
    )

    offs_c = tl.arange(0, BLOCK_C)

    # Pass 1: compute max over channels for numerical stability
    max_val = tl.full((), -float("inf"), dtype=tl.float32)
    for c_start in range(0, C, BLOCK_C):
        idx_c = c_start + offs_c
        mask = idx_c < C
        x_ptrs = x_ptr + base_offset + idx_c * stride_c
        x_vals = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        curr_max = tl.max(x_vals, axis=0)
        max_val = tl.maximum(max_val, curr_max)

    # Pass 2: compute sum of exp(x - max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for c_start in range(0, C, BLOCK_C):
        idx_c = c_start + offs_c
        mask = idx_c < C
        x_ptrs = x_ptr + base_offset + idx_c * stride_c
        x_vals = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        exp_vals = tl.exp(x_vals - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)

    inv_sum_exp = 1.0 / sum_exp

    # Pass 3: write sigmoid(softmax(x)) back
    for c_start in range(0, C, BLOCK_C):
        idx_c = c_start + offs_c
        mask = idx_c < C
        x_ptrs = x_ptr + base_offset + idx_c * stride_c
        x_vals = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        exp_vals = tl.exp(x_vals - max_val)
        softmax_vals = exp_vals * inv_sum_exp
        # Sigmoid
        sigmoid_vals = 1.0 / (1.0 + tl.exp(-softmax_vals))
        out_ptrs = out_ptr + base_offset + idx_c * stride_c
        tl.store(out_ptrs, sigmoid_vals, mask=mask)


def fused_softmax_sigmoid_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Fused Softmax(dim=1) + Sigmoid for a 5D tensor [N, C, D, H, W] using Triton.

    Args:
        x: Input tensor, expected shape [N, C, D, H, W], CUDA tensor.

    Returns:
        Tensor of same shape as x.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dim() == 5, "Input must have shape [N, C, D, H, W]"

    N, C, D, H, W = x.shape
    y = torch.empty_like(x)

    # Choose BLOCK_C as next power-of-2 >= C (and at least 1)
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    total_sites = N * D * H * W
    grid = (total_sites,)

    softmax_sigmoid_5d_kernel[grid](
        x, y,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + fused Softmax(dim=1) + Sigmoid (Triton).
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as native PyTorch (indexing is complex)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, C_in, D, H, W]
        x = self.conv_transpose(x)
        # x shape: [N, C_out, D, H, W]
        x = fused_softmax_sigmoid_3d(x)
        return x
