import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_fwd_kernel(
    x_ptr,        # *f32, [N, Ci, Di, Hi, Wi] contiguous
    w_ptr,        # *f32, [Co, Ci, Kd, Kh, Kw] contiguous
    b_ptr,        # *f32, [Co] (ignored if has_bias == False)
    out_ptr,      # *f32, [N, Co, Do, Ho, Wo] contiguous
    N, Ci, Di, Hi, Wi,
    Co, Do, Ho, Wo,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    Kd, Kh, Kw,
    M, K,         # M = N*Do*Ho*Wo, K = Ci*Kd*Kh*Kw
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < Co

    # Map linear M index -> (n, do, ho, wo)
    w_out = offs_m % Wo
    tmp = offs_m // Wo
    h_out = tmp % Ho
    tmp = tmp // Ho
    d_out = tmp % Do
    n_idx = tmp // Do

    n_2d = n_idx[:, None]
    d_out_2d = d_out[:, None]
    h_out_2d = h_out[:, None]
    w_out_2d = w_out[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Map linear K index -> (ci, kd, kh, kw)
        ci = offs_k // (Kd * Kh * Kw)
        rem = offs_k % (Kd * Kh * Kw)
        kd_idx = rem // (Kh * Kw)
        rem = rem % (Kh * Kw)
        kh_idx = rem // Kw
        kw_idx = rem % Kw

        # A tile (im2col input) [BM, BK]
        ci_a = ci[None, :]
        kd_a = kd_idx[None, :]
        kh_a = kh_idx[None, :]
        kw_a = kw_idx[None, :]

        z_in = d_out_2d * stride_d - pad_d + kd_a * dil_d
        y_in = h_out_2d * stride_h - pad_h + kh_a * dil_h
        x_in = w_out_2d * stride_w - pad_w + kw_a * dil_w

        in_bounds = (
            (z_in >= 0) & (z_in < Di) &
            (y_in >= 0) & (y_in < Hi) &
            (x_in >= 0) & (x_in < Wi)
        )

        a_mask = in_bounds & (mask_m[:, None]) & (k_mask[None, :])

        nci = n_2d * Ci + ci_a
        nd = nci * Di + z_in
        nh = nd * Hi + y_in
        nw = nh * Wi + x_in

        a_ptrs = x_ptr + nw
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile (weights) [BK, BN]
        ci_b = ci[:, None]
        kd_b = kd_idx[:, None]
        kh_b = kh_idx[:, None]
        kw_b = kw_idx[:, None]

        oc_2d = offs_n[None, :]
        b_mask = (k_mask[:, None]) & (mask_n[None, :])

        w_index = (((oc_2d * Ci + ci_b) * Kd + kd_b) * Kh + kh_b) * Kw + kw_b
        w_ptrs = w_ptr + w_index
        b = tl.load(w_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    if has_bias:
        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    # Map back to output tensor indices and store
    w_out = offs_m % Wo
    tmp = offs_m // Wo
    h_out = tmp % Ho
    tmp = tmp // Ho
    d_out = tmp % Do
    n_idx = tmp // Do

    n_2d = n_idx[:, None]
    d_out_2d = d_out[:, None]
    h_out_2d = h_out[:, None]
    w_out_2d = w_out[:, None]
    oc_2d = offs_n[None, :]

    out_index = (((n_2d * Co + oc_2d) * Do + d_out_2d) * Ho + h_out_2d) * Wo + w_out_2d
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_index, acc, mask=out_mask)


def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    # Fallback to PyTorch when Triton isn't applicable
    if (not x.is_cuda) or (not weight.is_cuda):
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
    if groups != 1:
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

    x = x.contiguous()
    weight = weight.contiguous()

    N, Ci, Di, Hi, Wi = x.shape
    Co, Ci_w, Kd, Kh, Kw = weight.shape
    assert Ci_w == Ci, "Incompatible in_channels between input and weight for groups=1."

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation

    Do = (Di + 2 * pad_d - dil_d * (Kd - 1) - 1) // stride_d + 1
    Ho = (Hi + 2 * pad_h - dil_h * (Kh - 1) - 1) // stride_h + 1
    Wo = (Wi + 2 * pad_w - dil_w * (Kw - 1) - 1) // stride_w + 1

    if Do <= 0 or Ho <= 0 or Wo <= 0:
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

    out = torch.empty((N, Co, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    M = N * Do * Ho * Wo
    K_total = Ci * Kd * Kh * Kw
    if M == 0 or Co == 0 or K_total == 0:
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(Co, meta["BLOCK_N"]),
    )

    has_bias = bias is not None
    bias_ptr = bias if bias is not None else x  # dummy pointer when has_bias is False

    conv3d_fwd_kernel[grid](
        x,
        weight,
        bias_ptr,
        out,
        N,
        Ci,
        Di,
        Hi,
        Wi,
        Co,
        Do,
        Ho,
        Wo,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w,
        Kd,
        Kh,
        Kw,
        M,
        K_total,
        has_bias=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return out


class ModelNew(nn.Module):
    """
    3D convolution implemented with a high-performance Triton kernel (groups=1 fast path).
    Falls back to PyTorch's conv3d for non-CUDA tensors, non-fp32 dtypes, or groups != 1.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        dilation: tuple = (1, 1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        # Use nn.Conv3d only for parameter management / initialization
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x,
            self.conv3d.weight,
            self.conv3d.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
