import torch, torch.nn as nn, triton, triton.language as tl


# ============================================
# 3D Convolution via GEMM-style Triton kernel
# ============================================
@triton.jit
def conv3d_gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, Cin, Co,
    D_in, H_in, W_in,
    kD, kH, kW,
    D_out, H_out, W_out,
    stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_w_co, stride_w_k,
    stride_y_n, stride_y_c, stride_y_d, stride_y_h, stride_y_w,
    M, K,
    BLOCK_M: tl.constexpr,  # tile over output positions (N * D_out * H_out * W_out)
    BLOCK_N: tl.constexpr,  # tile over output channels Co
    BLOCK_K: tl.constexpr,  # tile over reduction dim K = Cin * kD * kH * kW
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < Co

    # Decode flattened output position index offs_m -> (n, od, oh, ow)
    W_out_ = W_out
    H_out_ = H_out
    D_out_ = D_out

    tmp = offs_m // W_out_
    ow = offs_m % W_out_
    oh = tmp % H_out_
    tmp = tmp // H_out_
    od = tmp % D_out_
    n = tmp // D_out_

    # Base input pointer for each (n, od, oh, ow) at channel=0 and kernel offset (0,0,0)
    base_in = (
        x_ptr
        + n * stride_x_n
        + od * stride_x_d
        + oh * stride_x_h
        + ow * stride_x_w
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Decode flattened kernel index offs_k -> (ci, kd, kh, kw)
        tmp_k = offs_k // (kH * kW)
        kw = offs_k % kW
        kh = tmp_k % kH
        tmp_k = tmp_k // kH
        kd = tmp_k % kD
        ci = tmp_k // kD

        delta_k = (
            ci * stride_x_c
            + kd * stride_x_d
            + kh * stride_x_h
            + kw * stride_x_w
        )

        a_ptrs = base_in[:, None] + delta_k[None, :]
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )

        b_ptrs = w_ptr + offs_n[None, :] * stride_w_co + offs_k[:, None] * stride_w_k
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)
        k += BLOCK_K

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Store result to y[n, co, od, oh, ow]
    y_ptrs = (
        y_ptr
        + n[:, None] * stride_y_n
        + offs_n[None, :] * stride_y_c
        + od[:, None] * stride_y_d
        + oh[:, None] * stride_y_h
        + ow[:, None] * stride_y_w
    )
    tl.store(
        y_ptrs,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def conv3d_triton(x, weight, bias, kernel_size):
    """
    x:       (N, Cin, D_in, H_in, W_in)
    weight:  (Co, Cin, kD, kH, kW)
    bias:    (Co,)
    kernel_size: int or (kD, kH, kW)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dim() == 5 and weight.dim() == 5

    if isinstance(kernel_size, int):
        kD = kH = kW = kernel_size
    else:
        kD, kH, kW = kernel_size

    N, Cin, D_in, H_in, W_in = x.shape
    Co = weight.shape[0]

    D_out = D_in - kD + 1
    H_out = H_in - kH + 1
    W_out = W_in - kW + 1

    # Flatten kernel to (Co, K)
    K = Cin * kD * kH * kW
    w_flat = weight.contiguous().view(Co, K)

    # Output tensor
    y = torch.empty((N, Co, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    M = N * D_out * H_out * W_out

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(Co, META["BLOCK_N"]),
    )

    conv3d_gemm_kernel[grid](
        x, w_flat, bias, y,
        N, Cin, Co,
        D_in, H_in, W_in,
        kD, kH, kW,
        D_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        w_flat.stride(0), w_flat.stride(1),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3), y.stride(4),
        M, K,
        BLOCK_M=64, BLOCK_N=32, BLOCK_K=32,
    )
    return y


# ============================================
# Reduce-min along depth dimension (dim=2)
# ============================================
@triton.jit
def reduce_min_dim2_kernel(
    x_ptr, y_ptr,
    N, C, D, H, W,
    stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    BLOCK_D: tl.constexpr,
):
    # Each program handles one (n, c, h, w) and reduces over D
    pid = tl.program_id(0)
    total_rows = N * C * H * W

    row = pid
    mask_row = row < total_rows

    # Decode row -> (n, c, h, w)
    w_idx = row % W
    tmp = row // W
    h_idx = tmp % H
    tmp = tmp // H
    c_idx = tmp % C
    n_idx = tmp // C

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    mask = mask_row & mask_d

    base_x = (
        x_ptr
        + n_idx * stride_x_n
        + c_idx * stride_x_c
        + h_idx * stride_x_h
        + w_idx * stride_x_w
    )
    x_ptrs = base_x + offs_d * stride_x_d

    vals = tl.load(
        x_ptrs,
        mask=mask,
        other=float("inf"),
    )
    min_val = tl.min(vals, axis=0)

    base_y = (
        y_ptr
        + n_idx * stride_y_n
        + c_idx * stride_y_c
        + h_idx * stride_y_h
        + w_idx * stride_y_w
    )
    tl.store(base_y, min_val, mask=mask_row)


def reduce_min_dim2_triton(x, dim):
    """
    x: (N, C, D, H, W)
    dim must be 2 (depth).
    Returns y: (N, C, H, W) with min over D.
    """
    assert x.is_cuda
    assert x.dim() == 5
    assert dim == 2, "This Triton kernel reduces along dim=2 only."

    N, C, D, H, W = x.shape
    y = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(N * C * H * W, 1),)

    reduce_min_dim2_kernel[grid](
        x, y,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_D=32,
    )
    return y


# ============================================
# Softmax along channel dimension (dim=1) for (N, C, H, W)
# ============================================
@triton.jit
def softmax_dim1_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    BLOCK_C: tl.constexpr,
):
    # Each program computes softmax over channels for one (n, h, w)
    pid = tl.program_id(0)
    n_rows = N * H * W
    row = pid
    mask_row = row < n_rows

    # Decode row -> (n, h, w)
    w_idx = row % W
    tmp = row // W
    h_idx = tmp % H
    n_idx = tmp // H

    base_x = (
        x_ptr
        + n_idx * stride_x_n
        + h_idx * stride_x_h
        + w_idx * stride_x_w
    )
    base_y = (
        y_ptr
        + n_idx * stride_y_n
        + h_idx * stride_y_h
        + w_idx * stride_y_w
    )

    offs_c = tl.arange(0, BLOCK_C)

    # 1st pass: compute max over channels
    row_max = tl.full((1,), -float("inf"), dtype=tl.float32)
    c_start = 0
    while c_start < C:
        offs = c_start + offs_c
        mask_c = offs < C
        mask = mask_row & mask_c
        x_ptrs = base_x + offs * stride_x_c
        x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        chunk_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, chunk_max)
        c_start += BLOCK_C

    # 2nd pass: compute denominator (sum of exp)
    row_sum = tl.zeros((1,), dtype=tl.float32)
    c_start = 0
    while c_start < C:
        offs = c_start + offs_c
        mask_c = offs < C
        mask = mask_row & mask_c
        x_ptrs = base_x + offs * stride_x_c
        x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        x = x - row_max
        num = tl.exp(x)
        chunk_sum = tl.sum(num, axis=0)
        row_sum += chunk_sum
        c_start += BLOCK_C

    # 3rd pass: write normalized probabilities
    c_start = 0
    while c_start < C:
        offs = c_start + offs_c
        mask_c = offs < C
        mask = mask_row & mask_c
        x_ptrs = base_x + offs * stride_x_c
        x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        x = x - row_max
        num = tl.exp(x)
        out = num / row_sum
        y_ptrs = base_y + offs * stride_y_c
        tl.store(y_ptrs, out, mask=mask)
        c_start += BLOCK_C


def softmax_dim1_triton(x):
    """
    x: (N, C, H, W)
    Softmax along channel dimension C (dim=1).
    """
    assert x.is_cuda
    assert x.dim() == 4

    N, C, H, W = x.shape
    y = torch.empty_like(x)

    grid = lambda META: (triton.cdiv(N * H * W, 1),)

    softmax_dim1_kernel[grid](
        x, y,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_C=128,
    )
    return y


# ============================================
# ModelNew: Triton-accelerated replacement
# ============================================
class ModelNew(nn.Module):
    """
    Triton implementation of:
      - 3D convolution
      - min over a specified dimension (depth)
      - softmax over channel dimension
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        # Support int or tuple kernel_size, like nn.Conv3d
        if isinstance(kernel_size, int):
            kD = kH = kW = kernel_size
        else:
            kD, kH, kW = kernel_size
        self.kernel_size = (kD, kH, kW)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kD, kH, kW)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # 3D convolution
        x = conv3d_triton(x, self.weight, self.bias, self.kernel_size)
        # Min along depth (dim must be 2)
        x = reduce_min_dim2_triton(x, self.dim)
        # Softmax along channel dimension (dim=1)
        x = softmax_dim1_triton(x)
        return x
