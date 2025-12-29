import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def scale_maxpool3d_kernel(
    x_ptr, y_ptr,
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride_d, stride_h, stride_w,
    kernel_d, kernel_h, kernel_w,
    pad_d, pad_h, pad_w,
    stride_n_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
    stride_n_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out,
    scale,
    BLOCK_S: tl.constexpr,
):
    # Each program instance handles one (n, c) pair and a block of spatial output indices.
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)

    nc = pid_nc
    n = nc // C
    c = nc % C

    S = D_out * H_out * W_out

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_o = offs_s < S

    # Compute 3D output indices (d2, h2, w2) from flattened index
    w2 = offs_s % W_out
    tmp = offs_s // W_out
    h2 = tmp % H_out
    d2 = tmp // H_out

    # Corresponding input window start indices
    d0 = d2 * stride_d - pad_d
    h0 = h2 * stride_h - pad_h
    w0 = w2 * stride_w - pad_w

    # Base pointer for this (n, c) slice
    base_in = (
        x_ptr
        + n * stride_n_in
        + c * stride_c_in
        + d0 * stride_d_in
        + h0 * stride_h_in
        + w0 * stride_w_in
    )

    # Initialize max accumulator in FP32
    max_val = tl.full((BLOCK_S,), -float("inf"), dtype=tl.float32)

    # Loop over pooling window
    for kd in range(0, kernel_d):
        for kh in range(0, kernel_h):
            for kw in range(0, kernel_w):
                ptr = base_in + kd * stride_d_in + kh * stride_h_in + kw * stride_w_in
                val = tl.load(ptr, mask=mask_o, other=0.0)
                val = val.to(tl.float32)
                val = val * scale
                max_val = tl.maximum(max_val, val)

    # Store result
    base_out = (
        y_ptr
        + n * stride_n_out
        + c * stride_c_out
        + d2 * stride_d_out
        + h2 * stride_h_out
        + w2 * stride_w_out
    )
    tl.store(base_out, max_val.to(tl.float32), mask=mask_o)


def fused_scale_maxpool3d(x: torch.Tensor, scale: float, kernel_size):
    # x: (N, C, D, H, W), contiguous
    x = x.contiguous()
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dtype == torch.float32, "Kernels expect float32 tensors"

    if isinstance(kernel_size, int):
        k_d = k_h = k_w = kernel_size
    else:
        k_d, k_h, k_w = kernel_size

    # MaxPool3d defaults: stride = kernel_size, padding = 0, dilation = 1, ceil_mode = False
    stride_d = k_d
    stride_h = k_h
    stride_w = k_w
    pad_d = pad_h = pad_w = 0
    dilation_d = dilation_h = dilation_w = 1

    N, C, D_in, H_in, W_in = x.shape

    def _out_dim(L_in, k, s, p, d):
        # PyTorch pooling formula (ceil_mode=False)
        return (L_in + 2 * p - d * (k - 1) - 1) // s + 1

    D_out = _out_dim(D_in, k_d, stride_d, pad_d, dilation_d)
    H_out = _out_dim(H_in, k_h, stride_h, pad_h, dilation_h)
    W_out = _out_dim(W_in, k_w, stride_w, pad_w, dilation_w)

    y = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    stride_n_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in = x.stride()
    stride_n_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out = y.stride()

    grid = lambda META: (
        N * C,
        triton.cdiv(D_out * H_out * W_out, META["BLOCK_S"]),
    )

    scale_maxpool3d_kernel[grid](
        x, y,
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        k_d, k_h, k_w,
        pad_d, pad_h, pad_w,
        stride_n_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
        stride_n_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out,
        float(scale),
        BLOCK_S=128,
    )
    return y


@triton.jit
def global_avg_clamp3d_kernel(
    x_ptr, o_ptr,
    N, C,
    S,  # S = D * H * W
    stride_n_in, stride_c_in,
    stride_n_out, stride_c_out,
    clamp_min, clamp_max,
    BLOCK_S: tl.constexpr,
):
    # One program per (n, c) pair
    pid_nc = tl.program_id(0)
    nc = pid_nc
    n = nc // C
    c = nc % C

    base_in = x_ptr + n * stride_n_in + c * stride_c_in

    # Vector accumulator: sum across all spatial elements
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Loop over spatial elements in chunks of BLOCK_S
    for offset in range(0, S, BLOCK_S):
        idx = offset + tl.arange(0, BLOCK_S)
        mask = idx < S
        ptr = base_in + idx
        vals = tl.load(ptr, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        acc += vals

    total = tl.sum(acc, axis=0)
    mean = total / S

    # Clamp
    mean = tl.minimum(mean, clamp_max)
    mean = tl.maximum(mean, clamp_min)

    # Store to output (N, C, 1, 1, 1) contiguous: stride_w=1, stride_h=1, stride_d=1, stride_c=1, stride_n=C
    base_out = o_ptr + n * stride_n_out + c * stride_c_out
    tl.store(base_out, mean)


def global_avg_pool3d_clamp(x: torch.Tensor, clamp_min: float, clamp_max: float):
    # x: (N, C, D, H, W)
    x = x.contiguous()
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.dtype == torch.float32, "Kernels expect float32 tensors"

    N, C, D, H, W = x.shape
    S = D * H * W

    out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)

    stride_n_in, stride_c_in, _, _, _ = x.stride()
    stride_n_out, stride_c_out, _, _, _ = out.stride()

    grid = lambda META: (N * C,)

    global_avg_clamp3d_kernel[grid](
        x, out,
        N, C,
        S,
        stride_n_in, stride_c_in,
        stride_n_out, stride_c_out,
        float(clamp_min), float(clamp_max),
        BLOCK_S=256,
    )
    return out


class ModelNew(nn.Module):
    """
    Replacement model using Triton kernels for:
      - scalar multiply + MaxPool3d
      - global average pooling + clamp
    ConvTranspose3d is kept from PyTorch for correctness and simplicity.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.scale = float(scale)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0

    def forward(self, x):
        # x: (N, C_in, D, H, W)
        x = self.conv_transpose(x)          # (N, C_out, D1, H1, W1)
        x = fused_scale_maxpool3d(x, self.scale, self.maxpool_kernel_size)  # scaled + maxpooled
        x = global_avg_pool3d_clamp(x, self.clamp_min, self.clamp_max)      # global avg + clamp
        return x
