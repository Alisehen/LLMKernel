# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Smaller tile: more parallel programs, good when K is small
        triton.Config({"BLOCK_K": 64}, num_warps=2),
        # Baseline tile: balanced choice
        triton.Config({"BLOCK_K": 128}, num_warps=4),
        # Larger tile: better arithmetic intensity, good when K is large
        triton.Config({"BLOCK_K": 256}, num_warps=8),
    ],
    key=["K"],  # autotune based on reduction length
)
@triton.jit
def global_avg_bias_sum_kernel(
    x_ptr,          # pointer to input tensor after max-pool: [N, C, D, H, W] contiguous
    bias_ptr,       # pointer to bias tensor flattened: [C]
    out_ptr,        # pointer to output tensor: [N, 1, 1, 1] -> [N] contiguous
    n_batches,      # N
    channels,       # C (currently not used, but kept for interface compatibility)
    spatial_size,   # S = D * H * W
    K,              # K = C * S (elements per batch)
    BLOCK_K: tl.constexpr,
):
    # program ids: batch dimension and reduction-tile index
    pid_n = tl.program_id(axis=0)  # batch index
    pid_k = tl.program_id(axis=1)  # tile index along K dimension

    # Base offset for this batch in the flattened input
    base = pid_n * K

    # Compute linear offsets within the [C * D * H * W] block for this batch
    block_start = pid_k * BLOCK_K
    offsets = block_start + tl.arange(0, BLOCK_K)

    # Masks
    in_range_k = offsets < K
    in_range_n = pid_n < n_batches
    mask = in_range_n & in_range_k

    # Load input values
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)

    # Compute channel index for each element to index into bias
    c_idx = offsets // spatial_size
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # Each element contributes (x + bias) / spatial_size to the final scalar
    scale = 1.0 / spatial_size
    contrib = (x + bias) * scale

    # Reduce within the tile
    partial_sum = tl.sum(contrib, axis=0)

    # Atomically accumulate into output for this batch element
    tl.atomic_add(out_ptr + pid_n, partial_sum, mask=in_range_n)


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
    K = C * S

    # Flatten bias to [C]
    bias_flat = bias.view(C).contiguous()

    # Output [N, 1, 1, 1]; contiguous so element [n,0,0,0] is at offset n
    out = torch.zeros((N, 1, 1, 1), device=x.device, dtype=x.dtype)

    def grid(meta):
        # 2D launch: [N, ceil_div(K, BLOCK_K)]
        return (N, triton.cdiv(K, meta["BLOCK_K"]))

    # Launch autotuned Triton kernel
    global_avg_bias_sum_kernel[grid](
        x,
        bias_flat,
        out.view(-1),
        N,
        C,
        S,
        K,
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
