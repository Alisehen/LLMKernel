# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        # Larger tiles for better compute intensity
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=2,
        ),
        # Balanced baseline
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        # Safe small fallback for very tight register budgets
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['P', 'C_out', 'K_total'],
)
@triton.jit
def conv3d_gemm_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    N, C_in, C_out,
    D_in, H_in, W_in,
    K_D, K_H, K_W,
    D_out, H_out, W_out,
    stride_d_conv, stride_h_conv, stride_w_conv,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wd, stride_wh, stride_ww,  # kept for signature compat
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    P, K_total,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Program IDs along M (flattened N*D*H*W) and N (C_out)
    # -------------------------------------------------------------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < P
    mask_n = offs_n < C_out

    # -------------------------------------------------------------------------
    # Decode flattened M index -> (n, od, oh, ow)
    # -------------------------------------------------------------------------
    DHW = D_out * H_out * W_out
    HW = H_out * W_out

    n_idx = offs_m // DHW
    rem = offs_m % DHW
    od_idx = rem // HW
    rem = rem % HW
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Cast to int32 for cheaper address arithmetic
    n_idx = n_idx.to(tl.int32)
    od_idx = od_idx.to(tl.int32)
    oh_idx = oh_idx.to(tl.int32)
    ow_idx = ow_idx.to(tl.int32)

    # -------------------------------------------------------------------------
    # Precompute base offset into x for each M element (depends only on M tile)
    # -------------------------------------------------------------------------
    base_x_ptrs = (
        n_idx * stride_xn
        + od_idx * stride_d_conv * stride_xd
        + oh_idx * stride_h_conv * stride_xh
        + ow_idx * stride_w_conv * stride_xw
    )
    base_x_ptrs = base_x_ptrs.to(tl.int32)

    # Flatten kernel dims for fast index math
    K_hw = K_H * K_W
    K_dhw = K_D * K_hw

    # Precompute OC offsets for weight / output
    oc_offs = (offs_n * stride_wo).to(tl.int32)

    # -------------------------------------------------------------------------
    # Accumulator in FP32, stays in registers until final store
    # -------------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # GEMM-style loop over K dimension
    # -------------------------------------------------------------------------
    for k0 in range(0, K_total, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_total

        # Decompose flattened K index -> (ic, kd, kh, kw)
        ic_idx = offs_k // K_dhw
        remk = offs_k % K_dhw
        kd_idx = remk // K_hw
        remk2 = remk % K_hw
        kh_idx = remk2 // K_W
        kw_idx = remk2 % K_W

        ic_idx = ic_idx.to(tl.int32)
        kd_idx = kd_idx.to(tl.int32)
        kh_idx = kh_idx.to(tl.int32)
        kw_idx = kw_idx.to(tl.int32)

        # ------------------------- Load X tile ------------------------------
        # x_ptrs: [BLOCK_M, BLOCK_K]
        x_ptrs = (
            x_ptr
            + base_x_ptrs[:, None]
            + ic_idx[None, :] * stride_xc
            + kd_idx[None, :] * stride_xd
            + kh_idx[None, :] * stride_xh
            + kw_idx[None, :] * stride_xw
        )

        x_mask = mask_m[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # ------------------------- Load W tile ------------------------------
        # Treat weight as [K_total, C_out] with B[k, oc] = w[oc, k]
        # w_ptrs: [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + oc_offs[None, :] + offs_k[:, None]
        w_mask = k_mask[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # --------------------- FMA accumulation -----------------------------
        acc += tl.dot(x_vals, w_vals)

    # -------------------------------------------------------------------------
    # Fused bias add in registers
    # -------------------------------------------------------------------------
    bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    bias_vals = bias_vals.to(tl.float32)
    acc += bias_vals[None, :]

    # -------------------------------------------------------------------------
    # Single final store: y[n, oc, od, oh, ow]
    # -------------------------------------------------------------------------
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :].to(tl.int32) * stride_yc
        + od_idx[:, None] * stride_yd
        + oh_idx[:, None] * stride_yh
        + ow_idx[:, None] * stride_yw
    )
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def conv3d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                  stride=(1, 1, 1)) -> torch.Tensor:
    """
    High-performance Conv3d implementation in Triton, with fused bias add.
    Layout:
      x:      [N, C_in, D_in, H_in, W_in]
      weight: [C_out, C_in, K_D, K_H, K_W] (must be contiguous)
      bias:   [C_out]
    stride: (stride_d, stride_h, stride_w)
    padding=0, dilation=1, groups=1.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == weight.dtype == bias.dtype == torch.float32, "Use float32 tensors"

    if not weight.is_contiguous():
        weight = weight.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_w, K_D, K_H, K_W = weight.shape
    assert C_in == C_in_w, "Input channel mismatch"

    stride_d, stride_h, stride_w = stride

    # No padding, dilation=1
    D_out = (D_in - K_D) // stride_d + 1
    H_out = (H_in - K_H) // stride_h + 1
    W_out = (W_in - K_W) // stride_w + 1

    y = torch.empty((N, C_out, D_out, H_out, W_out),
                    device=x.device, dtype=x.dtype)

    x_strides = x.stride()
    w_strides = weight.stride()
    y_strides = y.stride()

    # Flattened dims
    P = N * D_out * H_out * W_out            # M dimension
    K_total = C_in * K_D * K_H * K_W         # K dimension

    def grid(META):
        return (
            triton.cdiv(P, META['BLOCK_M']),
            triton.cdiv(C_out, META['BLOCK_N']),
        )

    conv3d_gemm_kernel[grid](
        x, weight, bias, y,
        N, C_in, C_out,
        D_in, H_in, W_in,
        K_D, K_H, K_W,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3], x_strides[4],
        w_strides[0], w_strides[1], w_strides[2], w_strides[3], w_strides[4],
        y_strides[0], y_strides[1], y_strides[2], y_strides[3], y_strides[4],
        P, K_total,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-based replacement for the original Model:
      Conv3d (Triton) -> min along dim -> softmax along channel dim=1.
    Conv3d settings: stride=1, padding=0, dilation=1, groups=1.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            k_d = k_h = k_w = kernel_size
        else:
            assert len(kernel_size) == 3
            k_d, k_h, k_w = kernel_size
        self.kernel_size = (k_d, k_h, k_w)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        # Parameters matching nn.Conv3d layout
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, k_d, k_h, k_w))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Initialize similar to nn.Conv3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C_in, D, H, W]
        returns: softmax over channels after min along self.dim.
        """
        x = conv3d_triton(x, self.weight, self.bias, stride=(1, 1, 1))
        x, _ = torch.min(x, dim=self.dim)
        x = torch.softmax(x, dim=1)
        return x
