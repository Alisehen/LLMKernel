import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def compute_u_kernel(
    x_ptr,      # float32[B, Cin, H_in, W_in]
    u_ptr,      # float32[B, Cin, K_h, K_w]
    B,          # batch size (runtime)
    H_in, W_in,
    H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_ub, stride_uci, stride_ukh, stride_ukw,
    Cin: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Stage 1: For each (b, ci), compute

        U[b, ci, kh, kw] = sum_{hi, wi} x[b, ci, hi, wi] *
                           I_h(hi, kh) * I_w(wi, kw)

    where:
        ho = hi * stride_h - pad_h + kh
        wo = wi * stride_w - pad_w + kw
        I_h(hi, kh) = 1 if 0 <= ho < H_out else 0
        I_w(wi, kw) = 1 if 0 <= wo < W_out else 0

    This is the spatially aggregated contribution of input channel ci
    for kernel tap (kh, kw), independent of output channels.
    """
    pid_b = tl.program_id(0)
    pid_ci = tl.program_id(1)

    # Guard (in case grid is larger than B or Cin)
    if pid_b >= B or pid_ci >= Cin:
        return

    # Accumulator over kernel taps: [K_h, K_w]
    acc = tl.zeros((KERNEL_H, KERNEL_W), dtype=tl.float32)

    # Kernel offset vectors
    offs_kh = tl.arange(0, KERNEL_H)
    offs_kw = tl.arange(0, KERNEL_W)

    # Loop over input height
    for hi in range(0, H_IN):
        # Row accumulator over kw for this hi: [K_w]
        row_acc_kw = tl.zeros((KERNEL_W,), dtype=tl.float32)

        # Base pointer for this input row
        row_ptr = (
            x_ptr
            + pid_b * stride_xb
            + pid_ci * stride_xc
            + hi * stride_xh
        )

        # Tile over width
        for w_start in range(0, W_IN, BLOCK_W):
            offs_w = w_start + tl.arange(0, BLOCK_W)
            mask_w = offs_w < W_in

            x_ptrs = row_ptr + offs_w * stride_xw
            x_vals = tl.load(x_ptrs, mask=mask_w, other=0.0)  # [BLOCK_W]

            # Compute contributions for all kw in a vectorized way.
            # wi: [1, BLOCK_W]
            wi = offs_w[None, :]
            # kw: [K_w, 1]
            kw2d = offs_kw[:, None]

            # wo = wi * stride_w - pad_w + kw, shape [K_w, BLOCK_W]
            wo = wi * stride_w - pad_w + kw2d

            # Valid width mask: [K_w, BLOCK_W]
            valid_w = (wo >= 0) & (wo < W_out) & (mask_w[None, :])

            # Broadcast x_vals to [1, BLOCK_W] -> [K_w, BLOCK_W]
            x2d = x_vals[None, :]

            contrib = tl.where(valid_w, x2d, 0.0)  # [K_w, BLOCK_W]
            # Sum over width dimension -> [K_w]
            row_acc_kw += tl.sum(contrib, axis=1)

        # Now combine along height dimension using I_h(hi, kh)
        ho_base = hi * stride_h - pad_h
        ho = ho_base + offs_kh  # [K_h]
        valid_h = (ho >= 0) & (ho < H_out)
        valid_h_f = valid_h.to(tl.float32)  # [K_h]

        # Outer product: [K_h, 1] * [1, K_w] -> [K_h, K_w]
        valid_h_2d = valid_h_f[:, None]
        row_acc_2d = row_acc_kw[None, :]

        acc += valid_h_2d * row_acc_2d

    # Store U[b, ci, :, :]
    base_u_ptr = u_ptr + pid_b * stride_ub + pid_ci * stride_uci
    offs_kh_2d = offs_kh[:, None]
    offs_kw_2d = offs_kw[None, :]
    u_ptrs = base_u_ptr + offs_kh_2d * stride_ukh + offs_kw_2d * stride_ukw
    tl.store(u_ptrs, acc)


@triton.jit
def contract_u_w_kernel(
    u_ptr,      # float32[B, Cin, K_h, K_w]
    w_ptr,      # float32[Cin, Cout, K_h, K_w]
    sum_ptr,    # float32[B, Cout] (spatially summed conv output, no bias)
    B, Cout,
    stride_ub, stride_uci, stride_ukh, stride_ukw,
    stride_wci, stride_wco, stride_wkh, stride_wkw,
    stride_sum_b, stride_sum_c,
    Cin: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """
    Stage 2: Contract U with weights to get the global spatial sums
    over output channels:

        S[b, co] = sum_{ci, kh, kw} U[b, ci, kh, kw] * w[ci, co, kh, kw]
    """
    pid_b = tl.program_id(0)
    pid_co_block = tl.program_id(1)

    if pid_b >= B:
        return

    offs_co = pid_co_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
    mask_co = offs_co < Cout

    # Accumulator over output channel block
    acc_co = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels and kernel taps
    for ci in range(0, Cin):
        # Base pointer for U[b, ci, :, :]
        base_u_ptr = u_ptr + pid_b * stride_ub + ci * stride_uci

        for kh in range(0, KERNEL_H):
            for kw in range(0, KERNEL_W):
                # Load scalar U[b, ci, kh, kw]
                u_val_ptr = base_u_ptr + kh * stride_ukh + kw * stride_ukw
                u_val = tl.load(u_val_ptr)

                # Load weight vector w[ci, offs_co, kh, kw]
                w_ptrs = (
                    w_ptr
                    + ci * stride_wci
                    + offs_co * stride_wco
                    + kh * stride_wkh
                    + kw * stride_wkw
                )
                w_vals = tl.load(w_ptrs, mask=mask_co, other=0.0)

                acc_co += w_vals * u_val

    # Store the accumulated spatial sums
    sum_ptrs = sum_ptr + pid_b * stride_sum_b + offs_co * stride_sum_c
    tl.store(sum_ptrs, acc_co, mask=mask_co)


@triton.jit
def avg_affine_kernel(
    sum_ptr,      # float32[B, C]
    bias_ptr,     # float32[C]
    out_ptr,      # float32[B, C]
    B, C,
    spatial_size,
    multiplier,
    stride_sum_b, stride_sum_c,
    stride_out_b, stride_out_c,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = B * C
    mask = offs < total

    b = offs // C
    c = offs - b * C

    sum_ptrs = sum_ptr + b * stride_sum_b + c * stride_sum_c
    s = tl.load(sum_ptrs, mask=mask, other=0.0)

    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)

    # First global average pooling, then add bias, then multiply by scalar.
    val = multiplier * (s / spatial_size + bias)

    out_ptrs = out_ptr + b * stride_out_b + c * stride_out_c
    tl.store(out_ptrs, val, mask=mask)


def conv_transpose2d_global_avg_triton(x, weight, bias, stride, padding, output_padding, multiplier):
    """
    Computes:
        y = conv_transpose2d(x, weight, bias, stride, padding, output_padding)
        y = y * multiplier
        y = global_avg_pool(y)  # over H_out, W_out
        y = global_avg_pool(y)  # over spatial 1x1 => no-op

    Returns tensor of shape [B, C_out, 1, 1].

    Implementation uses an optimized two-stage algorithm:
      1) Aggregate input contributions per (b, ci, kh, kw) independently of output channels.
      2) Contract those aggregates with weights to get spatial sums per (b, co).
      3) Apply global average pooling, bias, and multiplier.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32

    B, Cin, H_in, W_in = x.shape
    Cin_w, Cout, K_h, K_w = weight.shape
    assert Cin_w == Cin, "Weight in_channels must match input channels"

    # Normalize stride/padding/output_padding to 2D tuples
    def _to_pair(v):
        if isinstance(v, int):
            return (v, v)
        return v

    stride_h, stride_w = _to_pair(stride)
    pad_h, pad_w = _to_pair(padding)
    out_pad_h, out_pad_w = _to_pair(output_padding)

    # Compute output spatial size (PyTorch conv_transpose2d formula, dilation=1)
    H_out = (H_in - 1) * stride_h - 2 * pad_h + (K_h - 1) + out_pad_h + 1
    W_out = (W_in - 1) * stride_w - 2 * pad_w + (K_w - 1) + out_pad_w + 1

    # Stage 1: compute U[b, ci, kh, kw]
    U = torch.empty((B, Cin, K_h, K_w), device=x.device, dtype=torch.float32)

    stride_xb, stride_xc, stride_xh, stride_xw = x.stride()
    stride_ub, stride_uci, stride_ukh, stride_ukw = U.stride()

    BLOCK_W = 64  # tile width for stage-1 reduction

    grid1 = (B, Cin)
    compute_u_kernel[grid1](
        x,
        U,
        B,
        H_in,
        W_in,
        H_out,
        W_out,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        stride_xb,
        stride_xc,
        stride_xh,
        stride_xw,
        stride_ub,
        stride_uci,
        stride_ukh,
        stride_ukw,
        Cin=Cin,
        KERNEL_H=K_h,
        KERNEL_W=K_w,
        H_IN=H_in,
        W_IN=W_in,
        BLOCK_W=BLOCK_W,
    )

    # Stage 2: contract U with weights to get spatial sums S_conv[b, co]
    sum_conv = torch.empty((B, Cout), device=x.device, dtype=torch.float32)

    stride_wci, stride_wco, stride_wkh, stride_wkw = weight.stride()
    stride_sum_b, stride_sum_c = sum_conv.stride()

    BLOCK_CO = 64
    grid2 = (B, triton.cdiv(Cout, BLOCK_CO))

    contract_u_w_kernel[grid2](
        U,
        weight,
        sum_conv,
        B,
        Cout,
        stride_ub,
        stride_uci,
        stride_ukh,
        stride_ukw,
        stride_wci,
        stride_wco,
        stride_wkh,
        stride_wkw,
        stride_sum_b,
        stride_sum_c,
        Cin=Cin,
        KERNEL_H=K_h,
        KERNEL_W=K_w,
        BLOCK_CO=BLOCK_CO,
    )

    # Stage 3: apply global average pooling, bias, and multiplier
    out = torch.empty_like(sum_conv)

    stride_out_b, stride_out_c = out.stride()
    spatial_size = H_out * W_out

    BLOCK = 128
    grid3 = (triton.cdiv(B * Cout, BLOCK),)

    avg_affine_kernel[grid3](
        sum_conv,
        bias,
        out,
        B,
        Cout,
        spatial_size,
        float(multiplier),
        stride_sum_b,
        stride_sum_c,
        stride_out_b,
        stride_out_c,
        BLOCK=BLOCK,
    )

    # Second global average pooling over 1x1 is a no-op.
    return out.view(B, Cout, 1, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()

        # Normalize kernel_size/stride/padding/output_padding
        def _to_pair(v):
            if isinstance(v, int):
                return (v, v)
            return v

        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.stride = _to_pair(stride)
        self.padding = _to_pair(padding)
        self.output_padding = _to_pair(output_padding)
        self.multiplier = float(multiplier)

        # Parameters to be mapped from original ConvTranspose2d
        # Weight layout matches nn.ConvTranspose2d: [in_channels, out_channels, k_h, k_w]
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, k_h, k_w))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Initialize similarly to nn.ConvTranspose2d (Kaiming uniform)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * k_h * k_w
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return conv_transpose2d_global_avg_triton(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.multiplier,
        )
