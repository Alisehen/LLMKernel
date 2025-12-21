# <corrected code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 128}, num_warps=8, num_stages=2),
    ],
    key=['channels', 'spatial_size'],
)
@triton.jit
def global_avg_bias_sum_kernel(
    x_ptr,          # pointer to input tensor: [N, C, D, H, W] contiguous
    bias_ptr,       # pointer to bias tensor flattened: [C]
    out_ptr,        # pointer to output tensor: [N] (flattened [N, 1, 1, 1])
    n_batches,      # N
    channels,       # C
    spatial_size,   # S = D * H * W
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(axis=0)  # batch index
    pid_cb = tl.program_id(axis=1)  # channel block index

    # which channels this program handles
    offs_c = pid_cb * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < channels

    valid_n = pid_n < n_batches

    # accumulator for sum over spatial dimension, per channel
    acc_c = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # common spatial offsets
    offs_s = tl.arange(0, BLOCK_S)

    # base offset for each (n, c) at spatial index 0
    # shape: [BLOCK_C]
    base_nc = (pid_n * channels + offs_c) * spatial_size

    # loop over spatial dimension in chunks of BLOCK_S
    s = 0
    while s < spatial_size:
        idx_s = s + offs_s  # [BLOCK_S]
        mask_s = idx_s < spatial_size  # [BLOCK_S]

        # 2D pointer: [BLOCK_C, BLOCK_S]
        ptr = x_ptr + base_nc[:, None] + idx_s[None, :]

        # combined mask for load
        mask = (mask_c[:, None] & mask_s[None, :] & valid_n)

        x = tl.load(ptr, mask=mask, other=0.0)

        # reduce over spatial axis (axis=1 => BLOCK_S)
        acc_c += tl.sum(x, axis=1).to(tl.float32)

        s += BLOCK_S

    # load bias for this block of channels
    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    # compute mean over spatial and add bias
    scale = 1.0 / spatial_size
    val_c = acc_c * scale + bias

    # zero-out contributions from invalid channels or batches
    val_c = tl.where(mask_c & valid_n, val_c, 0.0)

    # sum over channels in this block
    block_sum = tl.sum(val_c, axis=0)

    # accumulate into output for batch pid_n
    tl.atomic_add(out_ptr + pid_n, block_sum, mask=valid_n)


def triton_global_avg_bias_sum(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:    Tensor of shape [N, C, D, H, W] after conv, division, and max-pool.
    bias: Tensor of shape [C, 1, 1, 1] or [1, C, 1, 1, 1] (flattenable to [C]).

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

    # Grid: (batch, channel_blocks). Use lambda so autotune configs can change BLOCK_C.
    grid = lambda meta: (
        N,
        triton.cdiv(C, meta['BLOCK_C']),
    )

    global_avg_bias_sum_kernel[grid](
        x,
        bias_flat,
        out.view(-1),  # flatten to [N]
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
        self.sum_dim = sum_dim  # expected to be 1 in the original model

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
