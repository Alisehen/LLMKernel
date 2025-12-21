# <corrected code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64}, num_warps=2),
        triton.Config({"BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 256}, num_warps=8),
    ],
    key=["K"],  # autotune based on reduction length
)
@triton.jit
def global_avg_bias_sum_kernel(
    x_ptr,          # pointer to input tensor after max-pool: [N, C, D, H, W] contiguous, flattened
    bias_ptr,       # unused in kernel (bias is handled on host for correctness & efficiency)
    out_ptr,        # pointer to output partial sums: [N], one scalar per batch
    n_batches,      # N
    channels,       # C (kept for interface compatibility, not used in kernel)
    spatial_size,   # S = D * H * W (kept for interface compatibility, not used in kernel)
    K,              # K = C * S (elements per batch)
    BLOCK_K: tl.constexpr,
):
    # 2D launch: axis0 over batches, axis1 over reduction tiles
    pid_n = tl.program_id(axis=0)  # batch index
    pid_k = tl.program_id(axis=1)  # tile index along flattened [C * D * H * W]

    # Base offset for this batch in the flattened input
    base = pid_n * K

    # Compute linear offsets within the [C * D * H * W] block for this batch
    block_start = pid_k * BLOCK_K
    offsets = block_start + tl.arange(0, BLOCK_K)

    # Mask for valid reduction elements within this batch
    mask_k = offsets < K

    # Load input values; out-of-range elements are zeroed
    x = tl.load(x_ptr + base + offsets, mask=mask_k, other=0.0)

    # Reduce within the tile (sum over the BLOCK_K elements)
    partial_sum = tl.sum(x, axis=0)

    # Atomically accumulate into per-batch output sum (only sum of x; bias handled on host)
    tl.atomic_add(out_ptr + pid_n, partial_sum)


def triton_global_avg_bias_sum(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:    Tensor of shape [N, C, D, H, W] after conv, division, and max-pool.
    bias: Tensor broadcastable over [N, C, 1, 1, 1], with exactly C elements.

    Returns:
        Tensor of shape [N, 1, 1, 1] equal to:
        torch.sum( torch.nn.functional.adaptive_avg_pool3d(x, (1, 1, 1)) + bias, dim=1 )

    This is implemented as:
        out[n] = (1 / (D*H*W)) * x[n].sum() + bias.sum(dim=0).sum()
    which is algebraically equivalent to the expression above.
    """
    assert x.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    K = C * S

    # Flatten bias to [C] and ensure dtype matches x
    bias_flat = bias.view(-1)
    assert bias_flat.numel() == C, "Bias must contain exactly C elements"
    bias_flat = bias_flat.to(x.dtype).contiguous()

    # Output: per-batch sum of all elements (before dividing by S and adding bias)
    out_sum = torch.zeros((N,), device=x.device, dtype=x.dtype)

    def grid(meta):
        # 2D launch: [N, ceil_div(K, BLOCK_K)]
        return (N, triton.cdiv(K, meta["BLOCK_K"]))

    # Launch autotuned Triton kernel
    global_avg_bias_sum_kernel[grid](
        x,
        bias_flat,
        out_sum,
        N,
        C,
        S,
        K,
    )

    # Finalize on host:
    # out[n] = (1/S) * sum_{c,s} x[n,c,s] + sum_{c} bias[c]
    bias_sum = bias_flat.sum().to(x.dtype)
    out_sum = out_sum / S + bias_sum

    # Return as [N, 1, 1, 1]
    return out_sum.view(N, 1, 1, 1)


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
        # Kept for state_dict compatibility with the original model, but not used in forward
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim  # expected to be 1 (channels), handled in Triton path

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
