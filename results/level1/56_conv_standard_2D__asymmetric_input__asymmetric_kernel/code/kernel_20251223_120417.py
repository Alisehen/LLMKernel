import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def im2col_kernel(
    x_ptr,        # *T, [N, C_in, H, W]
    a_ptr,        # *T, [M, K_size] row-major (M=N*OH*OW, K_size=C_in_per_group*K_H*K_W)
    N, H, W,
    OH, OW,
    C_in_per_group,
    M, K_size,
    g,            # current group index
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # along rows (M = N*OH*OW)
    pid_k = tl.program_id(axis=1)  # along cols (K_size)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K_size

    # Decode m -> (n, oh, ow)
    ohow = OH * OW
    tmp = offs_m // ohow
    n_idx = tmp
    rem = offs_m - tmp * ohow
    oh_idx = rem // OW
    ow_idx = rem - oh_idx * OW

    # Decode k -> (ci_in_group, kh, kw)
    khkw = K_H * K_W
    ci = offs_k // khkw
    remk = offs_k - ci * khkw
    kh = remk // K_W
    kw = remk - kh * K_W

    # Broadcast
    n_b = n_idx[:, None]
    oh_b = oh_idx[:, None]
    ow_b = ow_idx[:, None]

    ci_b = ci[None, :]
    kh_b = kh[None, :]
    kw_b = kw[None, :]

    # Input spatial coordinates
    ih = oh_b * STRIDE_H + kh_b * DIL_H - PAD_H
    iw = ow_b * STRIDE_W + kw_b * DIL_W - PAD_W

    # Global input channel index for this group
    c_base = g * C_in_per_group
    c = ci_b + c_base

    # Bounds check
    mask_h = (ih >= 0) & (ih < H)
    mask_w = (iw >= 0) & (iw < W)
    mask_in = mask_m[:, None] & mask_k[None, :] & mask_h & mask_w

    # Load from input
    x_ptrs = (
        x_ptr
        + n_b * x_stride_n
        + c * x_stride_c
        + ih * x_stride_h
        + iw * x_stride_w
    )
    vals = tl.load(x_ptrs, mask=mask_in, other=0.0)

    # Store into A (row-major: stride_m = K_size, stride_k = 1)
    a_ptrs = a_ptr + offs_m[:, None] * K_size + offs_k[None, :]
    mask_out = mask_m[:, None] & mask_k[None, :]
    tl.store(a_ptrs, vals, mask=mask_out)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A tile [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + (
            offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        mask_a = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # B tile [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + (
            offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )
        mask_b = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    out = acc
    if IS_FP16:
        out = acc.to(tl.float16)
    if IS_BF16:
        out = acc.to(tl.bfloat16)

    c_ptrs = c_ptr + (
        offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, out, mask=mask_c)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
) -> torch.Tensor:
    assert x.ndim == 4, "Input must be NCHW"
    assert weight.ndim == 4, "Weight must be OIHW"

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    N, C_in, H, W = x.shape
    C_out, C_in_per_group, K_H, K_W = weight.shape
    assert C_in == C_in_per_group * groups, "Incompatible in_channels and groups"
    assert C_out % groups == 0, "out_channels must be divisible by groups"

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    # Output spatial size (PyTorch Conv2d formula)
    OH = (H + 2 * pad_h - dil_h * (K_H - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (K_W - 1) - 1) // stride_w + 1

    if OH <= 0 or OW <= 0:
        return x.new_empty((N, C_out, OH, OW))

    x_contig = x.contiguous()
    w_contig = weight.contiguous()

    # Dimensions for im2col + GEMM
    M = N * OH * OW
    C_out_per_group = C_out // groups
    K_size = C_in_per_group * K_H * K_W

    # Allocate im2col buffer A and output matrix C (row-major)
    A = torch.empty((M, K_size), device=x.device, dtype=x.dtype)
    C_mat = torch.empty((M, C_out), device=x.device, dtype=x.dtype)

    # Strides for input
    x_stride_n, x_stride_c, x_stride_h, x_stride_w = x_contig.stride()

    # Dtype flags for matmul epilogue
    is_fp16 = x.dtype == torch.float16
    is_bf16 = x.dtype == torch.bfloat16

    # Tiling parameters
    IM2COL_BLOCK_M = 64
    IM2COL_BLOCK_K = 64

    MM_BLOCK_M = 128
    MM_BLOCK_N = 64
    MM_BLOCK_K = 32

    def grid_im2col(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(K_size, meta["BLOCK_K"]),
        )

    def grid_mm(meta, m, n):
        return (
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]),
        )

    # Strides for A and C matrices (row-major)
    stride_am = K_size
    stride_ak = 1
    stride_cm = C_out
    stride_cn = 1

    for g in range(groups):
        # ---- im2col for this group ----
        im2col_kernel[grid_im2col](
            x_contig,
            A,
            N,
            H,
            W,
            OH,
            OW,
            C_in_per_group,
            M,
            K_size,
            g,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            STRIDE_H=stride_h,
            STRIDE_W=stride_w,
            PAD_H=pad_h,
            PAD_W=pad_w,
            DIL_H=dil_h,
            DIL_W=dil_w,
            K_H=K_H,
            K_W=K_W,
            BLOCK_M=IM2COL_BLOCK_M,
            BLOCK_K=IM2COL_BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # ---- Prepare weight matrix B_g of shape [K_size, C_out_per_group] ----
        w_g = w_contig[
            g * C_out_per_group : (g + 1) * C_out_per_group, :, :, :
        ]  # [C_out_per_group, C_in_per_group, K_H, K_W]
        B_g = w_g.view(C_out_per_group, K_size).transpose(0, 1).contiguous()
        # strides for B: row-major over K dimension
        stride_bk = C_out_per_group
        stride_bn = 1

        # Bias for this group
        if bias is not None:
            bias_g = bias[
                g * C_out_per_group : (g + 1) * C_out_per_group
            ]
            bias_ptr = bias_g
            has_bias = True
        else:
            # dummy pointer (never read if HAS_BIAS=False)
            bias_ptr = C_mat
            has_bias = False

        # Output slice for this group (columns subset)
        C_g_ptr = C_mat[:, g * C_out_per_group : (g + 1) * C_out_per_group]

        # ---- GEMM: [M, K_size] x [K_size, C_out_per_group] -> [M, C_out_per_group] ----
        matmul_kernel[grid_mm](
            A,
            B_g,
            bias_ptr,
            C_g_ptr,
            M,
            C_out_per_group,
            K_size,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            HAS_BIAS=has_bias,
            IS_FP16=is_fp16,
            IS_BF16=is_bf16,
            BLOCK_M=MM_BLOCK_M,
            BLOCK_N=MM_BLOCK_N,
            BLOCK_K=MM_BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

    # Reshape C_mat back to NCHW: (N, C_out, OH, OW)
    y = C_mat.view(N, OH, OW, C_out).permute(0, 3, 1, 2).contiguous()
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated 2D convolution layer using an im2col + GEMM formulation.

    Args mirror nn.Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        assert in_channels % groups == 0, "in_channels must be divisible by groups"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        kH, kW = kernel_size
        weight = torch.empty(
            out_channels,
            in_channels // groups,
            kH,
            kW,
        )

        # Kaiming uniform init, matching nn.Conv2d
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            bias_param = torch.empty(out_channels)
            fan_in = in_channels * kH * kW // groups
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias_param, -bound, bound)
            self.bias = nn.Parameter(bias_param)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
