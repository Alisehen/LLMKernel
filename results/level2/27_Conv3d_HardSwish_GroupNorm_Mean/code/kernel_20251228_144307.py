import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hardswish_groupnorm_kernel(
    x_ptr,        # (B, C, D, H, W)
    weight_ptr,   # (C,)
    bias_ptr,     # (C,)
    y_ptr,        # (B, C, D, H, W)
    B, C, D, H, W,
    num_groups,
    eps,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # range: [0, B * num_groups)
    # Map to (b, g)
    b = pid // num_groups
    g = pid % num_groups

    group_size = C // num_groups
    total_spatial = D * H * W
    L = group_size * total_spatial  # number of elements in this (b, g) group

    # Base pointers for this sample/group
    c0 = g * group_size
    x_base = x_ptr + b * stride_n + c0 * stride_c
    y_base = y_ptr + b * stride_n + c0 * stride_c

    offs = tl.arange(0, BLOCK_SIZE)
    sum_hs = 0.0
    sum_hs_sq = 0.0

    # ---- First pass: compute mean and variance of HardSwish(x) over the group ----
    for offset in range(0, L, BLOCK_SIZE):
        idx = offset + offs
        mask = idx < L

        # Decompose idx into (channel_offset, d, h, w)
        ch_off = idx // total_spatial
        rem = idx % total_spatial
        d_idx = rem // (H * W)
        rem2 = rem % (H * W)
        h_idx = rem2 // W
        w_idx = rem2 % W

        # Global channel index
        c_idx = c0 + ch_off

        x_ptrs = x_base + ch_off * stride_c + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # HardSwish: x * clamp(x + 3, 0, 6) / 6
        t = x + 3.0
        t = tl.minimum(t, 6.0)
        t = tl.maximum(t, 0.0)
        hs = x * t * (1.0 / 6.0)

        sum_hs += tl.sum(hs, axis=0)
        sum_hs_sq += tl.sum(hs * hs, axis=0)

    L_f = tl.float32(L)
    mean = sum_hs / L_f
    mean_sq = sum_hs_sq / L_f
    var = mean_sq - mean * mean
    var = tl.maximum(var, 0.0)  # numerical stability
    inv_std = 1.0 / tl.sqrt(var + eps)

    # ---- Second pass: normalize and apply affine (gamma, beta) ----
    for offset in range(0, L, BLOCK_SIZE):
        idx = offset + offs
        mask = idx < L

        ch_off = idx // total_spatial
        rem = idx % total_spatial
        d_idx = rem // (H * W)
        rem2 = rem % (H * W)
        h_idx = rem2 // W
        w_idx = rem2 % W

        c_idx = c0 + ch_off

        x_ptrs = x_base + ch_off * stride_c + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # HardSwish again
        t = x + 3.0
        t = tl.minimum(t, 6.0)
        t = tl.maximum(t, 0.0)
        hs = x * t * (1.0 / 6.0)

        # GroupNorm
        hs_norm = (hs - mean) * inv_std

        gamma = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
        beta = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

        y_val = hs_norm * gamma + beta

        y_ptrs = y_base + ch_off * stride_c + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
        tl.store(y_ptrs, y_val, mask=mask)


def hardswish_groupnorm(x: torch.Tensor,
                        weight: torch.Tensor,
                        bias: torch.Tensor,
                        num_groups: int,
                        eps: float) -> torch.Tensor:
    """
    x: (B, C, D, H, W)
    weight, bias: (C,)
    """
    assert x.is_cuda, "Input must be CUDA tensor"
    B, C, D, H, W = x.shape
    y = torch.empty_like(x)

    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()

    BLOCK_SIZE = 256

    grid = lambda META: (B * num_groups,)

    hardswish_groupnorm_kernel[grid](
        x, weight, bias, y,
        B, C, D, H, W,
        num_groups, eps,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


@triton.jit
def spatial_mean_kernel(
    x_ptr,      # (B, C, D, H, W)
    out_ptr,    # (B, C)
    B, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    stride_on, stride_oc,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # range: [0, B*C)
    b = pid // C
    c = pid % C

    total_spatial = D * H * W
    offs = tl.arange(0, BLOCK_SIZE)

    x_base = x_ptr + b * stride_n + c * stride_c
    sum_val = 0.0

    for offset in range(0, total_spatial, BLOCK_SIZE):
        idx = offset + offs
        mask = idx < total_spatial

        d_idx = idx // (H * W)
        rem = idx % (H * W)
        h_idx = rem // W
        w_idx = rem % W

        x_ptrs = x_base + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(x, axis=0)

    total_f = tl.float32(total_spatial)
    mean = sum_val / total_f

    out_ptr_single = out_ptr + b * stride_on + c * stride_oc
    tl.store(out_ptr_single, mean)


def spatial_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Mean over spatial dims [2, 3, 4], returning (B, C)
    """
    assert x.is_cuda, "Input must be CUDA tensor"
    B, C, D, H, W = x.shape
    out = torch.empty((B, C), device=x.device, dtype=x.dtype)

    stride_n, stride_c, stride_d, stride_h, stride_w = x.stride()
    stride_on, stride_oc = out.stride()

    BLOCK_SIZE = 256
    grid = lambda META: (B * C,)

    spatial_mean_kernel[grid](
        x, out,
        B, C, D, H, W,
        stride_n, stride_c, stride_d, stride_h, stride_w,
        stride_on, stride_oc,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Conv3D -> HardSwish -> GroupNorm -> spatial mean,
    where everything after Conv3D is fused into high-performance Triton kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        # Conv3D via cuDNN / PyTorch
        x = self.conv(x)

        # Fused HardSwish + GroupNorm via Triton
        x = hardswish_groupnorm(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )

        # Mean over spatial dims via Triton
        x = spatial_mean(x)
        return x
