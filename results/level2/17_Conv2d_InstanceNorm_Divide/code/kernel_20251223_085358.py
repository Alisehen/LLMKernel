import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_instance_norm_div_fused_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W,
    C_out, K_h, K_w,
    H_out, W_out,
    eps, inv_div,
    stride_in_n, stride_in_c, stride_in_h, stride_in_w,
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_HW: tl.constexpr,   # spatial tile size (H_out * W_out)
    BLOCK_CO: tl.constexpr,   # output channels per program
    BLOCK_K: tl.constexpr,    # reduction size over C_in * K_h * K_w
):
    # Program ids:
    pid_n = tl.program_id(0)  # batch index
    pid_co = tl.program_id(1)  # block of output channels

    n = pid_n
    co_start = pid_co * BLOCK_CO
    offs_co = co_start + tl.arange(0, BLOCK_CO)
    mask_co = offs_co < C_out

    # Constants
    S = H_out * W_out                      # number of spatial positions
    Kc = C_in * K_h * K_w                  # reduction over (C_in, K_h, K_w)
    KH_KW = K_h * K_w

    # Base pointers for this batch element
    base_in_n = x_ptr + n * stride_in_n
    base_out_n = y_ptr + n * stride_out_n

    # Load bias once for this channel block
    bias = tl.load(b_ptr + offs_co, mask=mask_co, other=0.0)

    # ---- First pass: compute mean and variance over H_out x W_out ----
    sum_ = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    for hw_start in range(0, S, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < S

        oh = offs_hw // W_out
        ow = offs_hw % W_out

        # Convolution for this spatial tile: [BLOCK_HW, BLOCK_CO]
        acc = tl.zeros((BLOCK_HW, BLOCK_CO), dtype=tl.float32)

        for k_start in range(0, Kc, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < Kc

            ic = offs_k // KH_KW
            rem_k = offs_k % KH_KW
            kh = rem_k // K_w
            kw = rem_k % K_w

            # Input pointers: [BLOCK_HW, BLOCK_K]
            ic_mat = ic[None, :]
            oh_mat = oh[:, None] + kh[None, :]
            ow_mat = ow[:, None] + kw[None, :]

            in_ptrs = (
                base_in_n
                + ic_mat * stride_in_c
                + oh_mat * stride_in_h
                + ow_mat * stride_in_w
            )
            mask_in = mask_hw[:, None] & k_mask[None, :]
            x = tl.load(in_ptrs, mask=mask_in, other=0.0)

            # Weight pointers: [BLOCK_K, BLOCK_CO]
            ic_w = ic[:, None]
            kh_w = kh[:, None]
            kw_w = kw[:, None]
            oc_w = offs_co[None, :]

            w_ptrs = (
                w_ptr
                + oc_w * stride_w_oc
                + ic_w * stride_w_ic
                + kh_w * stride_w_kh
                + kw_w * stride_w_kw
            )
            mask_w = k_mask[:, None] & mask_co[None, :]
            w = tl.load(w_ptrs, mask=mask_w, other=0.0)

            acc += tl.dot(x, w, allow_tf32=True)

        # Add bias
        acc += bias[None, :]

        # Accumulate statistics over spatial positions
        tile_sum = tl.sum(acc, axis=0)
        tile_sum_sq = tl.sum(acc * acc, axis=0)

        sum_ += tile_sum
        sum_sq += tile_sum_sq

    S_f = tl.cast(S, tl.float32)
    mean = sum_ / S_f
    mean_sq = sum_sq / S_f
    var = mean_sq - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = inv_std * inv_div

    # ---- Second pass: recompute conv and write normalized / divided output ----
    for hw_start in range(0, S, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < S

        oh = offs_hw // W_out
        ow = offs_hw % W_out

        # Convolution for this spatial tile: [BLOCK_HW, BLOCK_CO]
        acc = tl.zeros((BLOCK_HW, BLOCK_CO), dtype=tl.float32)

        for k_start in range(0, Kc, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < Kc

            ic = offs_k // KH_KW
            rem_k = offs_k % KH_KW
            kh = rem_k // K_w
            kw = rem_k % K_w

            ic_mat = ic[None, :]
            oh_mat = oh[:, None] + kh[None, :]
            ow_mat = ow[:, None] + kw[None, :]

            in_ptrs = (
                base_in_n
                + ic_mat * stride_in_c
                + oh_mat * stride_in_h
                + ow_mat * stride_in_w
            )
            mask_in = mask_hw[:, None] & k_mask[None, :]
            x = tl.load(in_ptrs, mask=mask_in, other=0.0)

            ic_w = ic[:, None]
            kh_w = kh[:, None]
            kw_w = kw[:, None]
            oc_w = offs_co[None, :]

            w_ptrs = (
                w_ptr
                + oc_w * stride_w_oc
                + ic_w * stride_w_ic
                + kh_w * stride_w_kh
                + kw_w * stride_w_kw
            )
            mask_w = k_mask[:, None] & mask_co[None, :]
            w = tl.load(w_ptrs, mask=mask_w, other=0.0)

            acc += tl.dot(x, w, allow_tf32=True)

        # Add bias
        acc += bias[None, :]

        # Instance norm + division
        acc = (acc - mean[None, :]) * scale[None, :]

        # Store output
        oc_mat = offs_co[None, :]
        oh_mat = oh[:, None]
        ow_mat = ow[:, None]

        out_ptrs = (
            base_out_n
            + oc_mat * stride_out_c
            + oh_mat * stride_out_h
            + ow_mat * stride_out_w
        )
        mask_out = mask_hw[:, None] & mask_co[None, :]
        tl.store(out_ptrs, acc, mask=mask_out)


def conv2d_instance_norm_div_fused_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    divide_by: float,
) -> torch.Tensor:
    """
    Fused NCHW Conv2d (stride=1, padding=0, dilation=1, groups=1)
    + InstanceNorm2d (affine=False, track_running_stats=False)
    + division by a scalar.

    Matches:
        y = conv2d(x, weight, bias)
        y = InstanceNorm2d(y, eps=eps, affine=False)
        y = y / divide_by
    """
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape

    # No padding, stride=1
    H_out = H - K_h + 1
    W_out = W - K_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_HW = 64   # spatial tile
    BLOCK_CO = 32   # output channels per program
    BLOCK_K = 32    # reduction tile over C_in*K_h*K_w

    def grid(META):
        return (
            N,
            triton.cdiv(C_out, META["BLOCK_CO"]),
        )

    inv_div = 1.0 / float(divide_by)

    conv2d_instance_norm_div_fused_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W,
        C_out, K_h, K_w,
        H_out, W_out,
        eps, inv_div,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_HW=BLOCK_HW,
        BLOCK_CO=BLOCK_CO,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of:
      Conv2d -> InstanceNorm2d -> division by constant

    Behavior matches the PyTorch reference Model.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        # Keep nn.Conv2d and nn.InstanceNorm2d for parameter & state_dict compatibility
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)  # eps/source of config
        self.divide_by = float(divide_by)

    def forward(self, x):
        # x is expected to be on CUDA for Triton kernels
        y = conv2d_instance_norm_div_fused_triton(
            x,
            self.conv.weight,
            self.conv.bias,
            self.instance_norm.eps,
            self.divide_by,
        )
        return y
