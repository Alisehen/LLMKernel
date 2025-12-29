import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_conv_bn_scale_kernel(
    x_ptr,            # *f32,   [B, Cin, H, W]
    w_ptr,            # *f32,   [Cout, Cin, K, K]
    bias_ptr,         # *f32,   [Cout]
    bn_scale_ptr,     # *f32,   [Cout]  (alpha)
    bn_shift_ptr,     # *f32,   [Cout]  (beta')
    out_ptr,          # *f32,   [B, Cout, Ho, Wo]
    Ho, Wo,           # int32
    M, Kdim,          # int32,  M = B*Ho*Wo, Kdim = Cin*K*K
    Cout, K,          # int32
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_oc, stride_w_ic, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_M: tl.constexpr,  # tile size in "M" (N * Ho * Wo)
    BLOCK_N: tl.constexpr,  # tile size in Cout
    BLOCK_K: tl.constexpr,  # tile size in Kdim
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_init = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < Cout

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    nhw = Ho * Wo
    kk = K * K

    # Loop over Kdim (Cin * K * K)
    for k_start in range(0, Kdim, BLOCK_K):
        offs_k = k_start + offs_k_init
        mask_k = offs_k < Kdim

        # ----- A tile (input-im2col) : [BLOCK_M, BLOCK_K] -----
        # m decomposition: [BM,1]
        m_b = offs_m[:, None]
        n_idx = m_b // nhw
        rem_m = m_b - n_idx * nhw
        ho_idx = rem_m // Wo
        wo_idx = rem_m - ho_idx * Wo

        # k decomposition for A: [1,BK]
        k_a = offs_k[None, :]
        cin_a = k_a // kk
        rem_ka = k_a - cin_a * kk
        kh_a = rem_ka // K
        kw_a = rem_ka - kh_a * K

        h_in = ho_idx + kh_a
        w_in = wo_idx + kw_a

        x_ptrs = (
            x_ptr
            + n_idx * stride_x_n
            + cin_a * stride_x_c
            + h_in * stride_x_h
            + w_in * stride_x_w
        )

        a_mask = mask_m[:, None] & mask_k[None, :]

        a = tl.load(x_ptrs, mask=a_mask, other=0.0)

        # ----- B tile (weights) : [BLOCK_K, BLOCK_N] -----
        k_b = offs_k[:, None]
        out_b = offs_n[None, :]

        cin_b = k_b // kk
        rem_kb = k_b - cin_b * kk
        kh_b = rem_kb // K
        kw_b = rem_kb - kh_b * K

        w_ptrs = (
            w_ptr
            + out_b * stride_w_oc
            + cin_b * stride_w_ic
            + kh_b * stride_w_kh
            + kw_b * stride_w_kw
        )

        b_mask = mask_k[:, None] & mask_n[None, :]

        b = tl.load(w_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    # Add convolution bias per output channel
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Apply fused BatchNorm + scaling: y = alpha * conv_out + beta'
    bn_scale = tl.load(bn_scale_ptr + offs_n, mask=mask_n, other=0.0)
    bn_shift = tl.load(bn_shift_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * bn_scale[None, :] + bn_shift[None, :]

    # Store result to [B, Cout, Ho, Wo]
    m_b = offs_m[:, None]
    n_idx = m_b // nhw
    rem_m = m_b - n_idx * nhw
    ho_idx = rem_m // Wo
    wo_idx = rem_m - ho_idx * Wo

    out_ptrs = (
        out_ptr
        + n_idx * stride_out_n
        + offs_n[None, :] * stride_out_c
        + ho_idx * stride_out_h
        + wo_idx * stride_out_w
    )

    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def fused_conv_bn_scale(x, weight, bias, bn_scale, bn_shift, scaling_factor):
    # x:        [B, Cin, H, W]
    # weight:   [Cout, Cin, K, K]
    # bias:     [Cout]
    # bn_scale: [Cout]  (gamma / sqrt(var+eps) * scaling_factor)
    # bn_shift: [Cout]  ((beta - mu*gamma/sqrt(var+eps)) * scaling_factor)
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."
    assert x.dtype == torch.float32, "This fused kernel currently supports float32."

    B, Cin, H, W = x.shape
    Cout, Cin_w, K, K_w = weight.shape
    assert Cin == Cin_w and K == K_w, "Incompatible input/weight shapes."

    # Only stride=1, padding=0, dilation=1, groups=1 are handled
    Ho = H - K + 1
    Wo = W - K + 1
    assert Ho > 0 and Wo > 0, "Input too small for valid convolution with given kernel size."

    # Ensure contiguous layouts for best performance
    x_c = x.contiguous()
    w_c = weight.contiguous()
    bias_c = bias.contiguous()
    bn_scale_c = bn_scale.contiguous()
    bn_shift_c = bn_shift.contiguous()

    out = torch.empty((B, Cout, Ho, Wo), device=x.device, dtype=x.dtype)

    M = B * Ho * Wo
    Kdim = Cin * K * K

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(Cout, META["BLOCK_N"]),
    )

    fused_conv_bn_scale_kernel[grid](
        x_c, w_c, bias_c, bn_scale_c, bn_shift_c, out,
        Ho, Wo, M, Kdim, Cout, K,
        x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
        w_c.stride(0), w_c.stride(1), w_c.stride(2), w_c.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )

    return out


class ModelNew(nn.Module):
    """
    Fused Triton implementation of:
      Conv2d -> BatchNorm2d (inference) -> scaling_factor
    The BatchNorm is applied in inference mode using running_mean/var.
    """

    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        # Match original module structure for parameter compatibility
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        assert x.ndim == 4, "Input must be NCHW"
        assert self.conv.stride == (1, 1), "Kernel only supports stride=1"
        assert self.conv.padding == (0, 0), "Kernel only supports padding=0"
        assert self.conv.dilation == (1, 1), "Kernel only supports dilation=1"
        assert self.conv.groups == 1, "Kernel only supports groups=1"

        # Prepare parameters
        weight = self.conv.weight
        if self.conv.bias is None:
            bias = torch.zeros(
                weight.size(0),
                device=weight.device,
                dtype=weight.dtype,
            )
        else:
            bias = self.conv.bias

        bn = self.bn
        running_mean = bn.running_mean.to(weight.device, dtype=weight.dtype)
        running_var = bn.running_var.to(weight.device, dtype=weight.dtype)
        gamma = bn.weight.to(weight.device, dtype=weight.dtype)
        beta = bn.bias.to(weight.device, dtype=weight.dtype)
        eps = bn.eps

        # Precompute fused BatchNorm (inference) + scaling parameters
        # out = scaling_factor * [ (x - mu)/sqrt(var+eps) * gamma + beta ]
        inv_std = torch.rsqrt(running_var + eps)
        bn_scale = gamma * inv_std * self.scaling_factor
        bn_shift = (beta - running_mean * inv_std * gamma) * self.scaling_factor

        return fused_conv_bn_scale(x, weight, bias, bn_scale, bn_shift, self.scaling_factor)
