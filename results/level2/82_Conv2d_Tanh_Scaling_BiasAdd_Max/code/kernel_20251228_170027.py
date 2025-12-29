import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_tanh_scale_bias_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, y_ptr,
    M, N, K,                      # M = N_batch * OH * OW, N = out_channels, K = C * KH * KW
    N_batch, C, H, W,
    OH, OW, KH, KW,
    scaling_factor,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets in M (output positions flattened) and N (output channels)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # Compute (n, oh, ow) for each m in this tile
    ohw = OH * OW
    n_idx = offs_m // ohw
    rem = offs_m - n_idx * ohw
    oh = rem // OW
    ow = rem - oh * OW

    # Base input/output offsets for each (n, oh, ow)
    base_in = n_idx * stride_xn + oh * stride_xh + ow * stride_xw
    base_out = n_idx * stride_yn + oh * stride_yh + ow * stride_yw

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop over C * KH * KW
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Map k -> (ci, kh, kw)
        kk = offs_k
        k_per_c = KH * KW
        ci = kk // k_per_c
        remk = kk - ci * k_per_c
        kh = remk // KW
        kw = remk - kh * KW

        # Offsets for input and weight for each k
        offset_x_k = ci * stride_xc + kh * stride_xh + kw * stride_xw
        offset_w_k = ci * stride_wc + kh * stride_wkh + kw * stride_wkw

        # Pointers for A (input "im2col") tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = x_ptr + base_in[:, None] + offset_x_k[None, :]
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for B (weights) tile: shape (BLOCK_K, BLOCK_N)
        b_ptrs = w_ptr + offset_w_k[:, None] + offs_n[None, :] * stride_wn
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply accumulate
        acc += tl.dot(a, b, allow_tf32=True)

    # Add convolution bias (pre-activation)
    conv_b = tl.load(conv_bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + conv_b[None, :]

    # Tanh activation: tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    two_x = acc * 2.0
    e = tl.exp(two_x)
    acc = (e - 1.0) / (e + 1.0)

    # Scaling
    acc = acc * scaling_factor

    # Add extra bias (post-scaling)
    extra_b = tl.load(extra_bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + extra_b[None, :]

    # Store to output tensor in NCHW layout
    y_ptrs = y_ptr + base_out[:, None] + offs_n[None, :] * stride_yc
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def fused_conv2d_tanh_scale_bias(x, weight, conv_bias, extra_bias, scaling_factor):
    """
    x:        (N, C, H, W)
    weight:   (OC, C, KH, KW)
    conv_bias:(OC,)
    extra_bias: (OC,)  -- from bias_shape (OC,1,1) flattened
    scaling_factor: float
    """
    assert x.is_cuda and weight.is_cuda and conv_bias.is_cuda and extra_bias.is_cuda
    N_batch, C, H, W = x.shape
    OC, Cw, KH, KW = weight.shape
    assert C == Cw

    # Valid 2D conv: stride=1, padding=0, dilation=1
    OH = H - KH + 1
    OW = W - KW + 1

    M = N_batch * OH * OW
    K = C * KH * KW
    N_out = OC

    y = torch.empty((N_batch, OC, OH, OW), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wn, stride_wc, stride_wkh, stride_wkw = weight.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N_out, META["BLOCK_N"]),
    )

    conv2d_tanh_scale_bias_kernel[grid](
        x, weight, conv_bias, extra_bias, y,
        M, N_out, K,
        N_batch, C, H, W,
        OH, OW, KH, KW,
        scaling_factor,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wn, stride_wc, stride_wkh, stride_wkw,
        stride_yn, stride_yc, stride_yh, stride_yw,
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return y


@triton.jit
def max_pool2d_kernel(
    x_ptr, y_ptr,
    N_batch, C, H_in, W_in,
    H_out, W_out,
    kernel_h, kernel_w,
    stride_h, stride_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_P: tl.constexpr,
):
    # program_id(0): over N_batch * C (batch-channel)
    # program_id(1): over pooled spatial positions (H_out * W_out) in blocks
    pid_bc = tl.program_id(0)
    pid_p = tl.program_id(1)

    BC = N_batch * C
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    P = H_out * W_out

    bc_mask = pid_bc < BC
    p_mask = offs_p < P

    # Decode (n, c) from bc index
    n_idx = pid_bc // C
    c_idx = pid_bc - n_idx * C

    # Base offsets for this (n, c) in input / output
    base_x_bc = n_idx * stride_xn + c_idx * stride_xc
    base_y_bc = n_idx * stride_yn + c_idx * stride_yc

    # Decode (ph, pw) from flattened pooled position
    ph = offs_p // W_out
    pw = offs_p - ph * W_out

    # Top-left corner of pooling window in input
    h0 = ph * stride_h
    w0 = pw * stride_w

    # Initialize max values
    max_vals = tl.full((BLOCK_P,), -float("inf"), dtype=tl.float32)

    # Iterate over pooling window
    for kh in range(0, kernel_h):
        for kw in range(0, kernel_w):
            h = h0 + kh
            w = w0 + kw

            # Bounds check for input indices
            in_bounds = (h < H_in) & (w < W_in)
            mask = p_mask & bc_mask & in_bounds

            ptrs = x_ptr + base_x_bc + h * stride_xh + w * stride_xw
            vals = tl.load(ptrs, mask=mask, other=-float("inf"))
            vals_f32 = vals.to(tl.float32)
            max_vals = tl.maximum(max_vals, vals_f32)

    # Store results
    out_ptrs = y_ptr + base_y_bc + ph * stride_yh + pw * stride_yw
    store_mask = p_mask & bc_mask
    tl.store(out_ptrs, max_vals, mask=store_mask)


def fused_max_pool2d(x, kernel_size, stride=None):
    """
    x: (N, C, H_in, W_in)
    kernel_size: int
    stride: int or None (defaults to kernel_size)
    """
    assert x.is_cuda
    if stride is None:
        stride = kernel_size

    N_batch, C, H_in, W_in = x.shape
    KH = kernel_size
    KW = kernel_size
    SH = stride
    SW = stride

    # PyTorch MaxPool2d output size for no padding, no dilation
    H_out = (H_in - KH) // SH + 1
    W_out = (W_in - KW) // SW + 1

    y = torch.empty((N_batch, C, H_out, W_out), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    P = H_out * W_out
    grid = lambda META: (
        N_batch * C,
        triton.cdiv(P, META["BLOCK_P"]),
    )

    max_pool2d_kernel[grid](
        x, y,
        N_batch, C, H_in, W_in,
        H_out, W_out,
        KH, KW,
        SH, SW,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_yn, stride_yc, stride_yh, stride_yw,
        BLOCK_P=128,
    )
    return y


class ModelNew(nn.Module):
    """
    Optimized model using Triton kernels:
    Conv2d + tanh + scaling + bias addition (fused) followed by MaxPool2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        # Keep Conv2d module for parameter compatibility; we use its weights in Triton
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        # Ensure contiguous NCHW layout
        x = x.contiguous()

        # Conv2d + tanh + scaling + extra bias (fused Triton kernel)
        weight = self.conv.weight
        conv_bias = self.conv.bias
        extra_bias = self.bias.view(-1)
        x = fused_conv2d_tanh_scale_bias(
            x,
            weight,
            conv_bias,
            extra_bias,
            self.scaling_factor,
        )

        # MaxPool2d via Triton
        k = self.max_pool.kernel_size
        if isinstance(k, tuple):
            assert k[0] == k[1], "Only square kernels supported in this Triton implementation"
            k = k[0]
        s = self.max_pool.stride
        if s is None:
            s = k
        if isinstance(s, tuple):
            assert s[0] == s[1], "Only square strides supported in this Triton implementation"
            s = s[0]

        x = fused_max_pool2d(x, k, stride=s)
        return x
