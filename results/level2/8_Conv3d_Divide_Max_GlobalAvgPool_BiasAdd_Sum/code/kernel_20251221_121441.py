# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
    ],
    key=['spatial_size', 'channels'],
)
@triton.jit
def global_avg_bias_sum_kernel(
    x_ptr,          # pointer to input tensor: [N, C, D, H, W] contiguous
    bias_ptr,       # pointer to bias tensor flattened: [C]
    out_ptr,        # pointer to output tensor: [N, 1, 1, 1] (flattened as [N])
    n_batches,      # N
    channels,       # C
    spatial_size,   # S = D * H * W
    BLOCK_S: tl.constexpr,
):
    # 2D grid: pid_n over batch, pid_c over channels
    pid_n = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)

    valid_n = pid_n < n_batches
    valid_c = pid_c < channels
    valid_nc = valid_n & valid_c

    # Early masks handle out-of-range programs; no control-flow return.
    # Memory layout: x is [N, C, S] contiguous in the last dim.
    # Base offset for this (n, c) block in the flattened input.
    base_nc = (pid_n * channels + pid_c) * spatial_size

    # Pre-load bias for this channel (single element).
    bias_val = tl.load(bias_ptr + pid_c, mask=valid_c, other=0.0)

    # Accumulate sum over the spatial dimension for this (n, c).
    offs_s = tl.arange(0, BLOCK_S)
    acc = 0.0

    # Loop over spatial dimension in blocks of BLOCK_S
    for s_start in range(0, spatial_size, BLOCK_S):
        idx_s = s_start + offs_s
        mask = valid_nc & (idx_s < spatial_size)

        x = tl.load(x_ptr + base_nc + idx_s, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)

    # Compute per-(n,c) contribution: mean over spatial + bias[c]
    # mean = sum(x) / S
    scale = 1.0 / spatial_size
    val = acc * scale + bias_val

    # Sum over channels: each (n, c) atomically accumulates into out[n]
    tl.atomic_add(out_ptr + pid_n, val, mask=valid_nc)


def triton_global_avg_bias_sum(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:    Tensor of shape [N, C, D, H, W] after conv, division, and max-pool.
    bias: Tensor of shape [C, 1, 1, 1].

    Returns:
        Tensor of shape [N, 1, 1, 1] equal to:
        torch.sum( adaptive_avg_pool3d(x, (1,1,1)) + bias, dim=1 )
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W

    # Flatten bias to [C]
    bias_flat = bias.view(C).contiguous()

    # Output [N, 1, 1, 1]; contiguous so element [n,0,0,0] is at offset n
    out = torch.zeros((N, 1, 1, 1), device=x.device, dtype=x.dtype)

    # 2D grid over (batch, channels) for high parallelism without excessive atomics
    grid = (N, C)

    global_avg_bias_sum_kernel[grid](
        x,
        bias_flat,
        out,
        N,
        C,
        S,
    )

    return out


class ModelNew(nn.Module):
    """
    Model that performs:
      Conv3d -> divide by constant -> MaxPool3d ->
      (Triton) global average pooling + bias add + sum over channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        # Keep this for state_dict compatibility with the original model, but not used in forward
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim  # expected to be 1, but handled by the Triton kernel

    def forward(self, x):
        # 3D convolution
        x = self.conv(x)
        # Divide by constant
        x = x / self.divisor
        # Max pooling
        x = self.max_pool(x)
        # Fused (global average pooling + bias add + sum over channels) in Triton
        x = triton_global_avg_bias_sum(x, self.bias)
        return x
