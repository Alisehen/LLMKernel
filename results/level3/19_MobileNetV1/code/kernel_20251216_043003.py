import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------
# Triton Conv2D (NCHW) Kernel - Implicit GEMM, Grouped
# ----------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["N", "C_in", "H_in", "W_in", "C_out", "KH", "KW", "groups"],
)
@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    C_in_per_group, C_out_per_group,
    KH, KW,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    BLOCK_M: tl.constexpr,  # tiles over output positions (N * H_out * W_out)
    BLOCK_N: tl.constexpr,  # tiles over output channels per group
    BLOCK_K: tl.constexpr,  # tiles over K = C_in_per_group * KH * KW
):
    # --- Program IDs ---
    pid_g = tl.program_id(axis=0)  # group id
    pid_m = tl.program_id(axis=1)  # tile over output positions
    pid_n = tl.program_id(axis=2)  # tile over output channels within group

    # --- Dimensions ---
    HW_out = H_out * W_out
    M_total = N * HW_out
    K_total = C_in_per_group * KH * KW

    # --- Offsets for this program ---
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    # Each block processes one group
    co_base = pid_g * C_out_per_group
    co = co_base + offs_n  # global output channels
    mask_m = offs_m < M_total
    mask_n = offs_n < C_out_per_group
    mask_co = co < C_out

    # Decode output position m -> (n, oh, ow)
    n = offs_m // HW_out
    rem = offs_m % HW_out
    oh = rem // W_out
    ow = rem % W_out

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = C_in_per_group * KH * KW
    for k0 in range(0, K_total, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)
        k_idx = k0 + offs_k  # [BK]
        mask_k = k_idx < K_total

        # Map k_idx -> (ci, kh, kw) within group
        ci = k_idx % C_in_per_group
        tmp = k_idx // C_in_per_group
        kw = tmp % KW
        kh = tmp // KW

        # Global input channel index for this group
        ic = pid_g * C_in_per_group + ci  # [BK]

        # Input spatial coordinates for each (m, k)
        h_in = oh[:, None] * stride_h - pad_h + kh[None, :] * dilation_h
        w_in = ow[:, None] * stride_w - pad_w + kw[None, :] * dilation_w

        # Compute input pointers for A tile: [BM, BK]
        x_ptrs = (
            x_ptr
            + n[:, None] * stride_x_n
            + ic[None, :] * stride_x_c
            + h_in * stride_x_h
            + w_in * stride_x_w
        )

        # Validity mask for A tile
        mask_x = (
            mask_m[:, None]
            & mask_k[None, :]
            & (ic[None, :] < C_in)
            & (h_in >= 0)
            & (h_in < H_in)
            & (w_in >= 0)
            & (w_in < W_in)
        )

        a = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight pointers for B tile: [BK, BN]
        # w[co, ci, kh, kw]
        w_offset_k = (
            ci * stride_w_ci
            + kh * stride_w_kh
            + kw * stride_w_kw
        )  # [BK]
        w_ptrs = (
            w_ptr
            + w_offset_k[:, None]
            + co[None, :] * stride_w_co
        )

        mask_w = mask_k[:, None] & mask_n[None, :] & mask_co[None, :]
        b = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # GEMM on this K-tile
        acc += tl.dot(a, b, allow_tf32=True)

    # Store output Y
    y_ptrs = (
        y_ptr
        + n[:, None] * stride_y_n
        + co[None, :] * stride_y_c
        + oh[:, None] * stride_y_h
        + ow[:, None] * stride_y_w
    )
    mask_y = mask_m[:, None] & mask_n[None, :] & mask_co[None, :]
    tl.store(y_ptrs, acc, mask=mask_y)


# ----------------------------
# Triton Linear (GEMM) Kernel
# ----------------------------

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Base pointers for first K-tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        k_mask = k_idx < K

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused bias add with same N offsets as matmul/store
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


# ----------------------------
# Python Wrappers
# ----------------------------

def _pair(v):
    if isinstance(v, tuple):
        return v
    return (v, v)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> torch.Tensor:
    """
    x:      [N, C_in, H_in, W_in]
    weight: [C_out, C_in/groups, KH, KW]
    bias:   [C_out] or None
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_pg, KH, KW = weight.shape

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    assert C_in % groups == 0
    assert C_out % groups == 0
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups
    assert C_in_per_group == C_in_pg, "Weight shape inconsistent with groups and C_in"

    # Output dimensions (PyTorch-style)
    H_out = (H_in + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    y = torch.empty(
        (N, C_out, H_out, W_out), device=x.device, dtype=torch.float32
    )

    M_total = N * H_out * W_out

    def grid(META):
        return (
            groups,
            triton.cdiv(M_total, META["BLOCK_M"]),
            triton.cdiv(C_out_per_group, META["BLOCK_N"]),
        )

    conv2d_nchw_kernel[grid](
        x_contig, w_contig, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        groups,
        C_in_per_group, C_out_per_group,
        KH, KW,
        x_contig.stride(0), x_contig.stride(1),
        x_contig.stride(2), x_contig.stride(3),
        w_contig.stride(0), w_contig.stride(1),
        w_contig.stride(2), w_contig.stride(3),
        y.stride(0), y.stride(1),
        y.stride(2), y.stride(3),
    )

    if bias is not None:
        y += bias.view(1, -1, 1, 1)

    return y.to(x.dtype)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K]  (nn.Linear weight.shape)
    bias:   [N]
    Returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w

    a = x.contiguous()
    # A[M, K] @ B[K, N]; B is weight.T
    b = weight.t().contiguous()
    c = torch.empty((M, N), device=x.device, dtype=torch.float32)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N, META["BLOCK_N"]),
        )

    linear_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c.to(x.dtype)


# ----------------------------
# PyTorch Modules Using Triton Kernels
# ----------------------------

class TritonConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        k_h, k_w = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, k_h, k_w) * 0.01
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
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


class TritonLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / (in_features ** 0.5))
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_features]
        return triton_linear(x, self.weight, self.bias)


# ----------------------------
# MobileNetV1 with Triton Kernels
# ----------------------------

class ModelNew(nn.Module):
    """
    MobileNetV1 using Triton-based Conv2d and Linear.
    BatchNorm and ReLU are kept as PyTorch ops.
    """
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(ModelNew, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                TritonConv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                TritonConv2d(inp, inp, kernel_size=3, stride=stride, padding=1,
                             groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                TritonConv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        c32 = int(32 * alpha)
        c64 = int(64 * alpha)
        c128 = int(128 * alpha)
        c256 = int(256 * alpha)
        c512 = int(512 * alpha)
        c1024 = int(1024 * alpha)

        self.model = nn.Sequential(
            conv_bn(input_channels, c32, 2),
            conv_dw(c32, c64, 1),
            conv_dw(c64, c128, 2),
            conv_dw(c128, c128, 1),
            conv_dw(c128, c256, 2),
            conv_dw(c256, c256, 1),
            conv_dw(c256, c512, 2),
            conv_dw(c512, c512, 1),
            conv_dw(c512, c512, 1),
            conv_dw(c512, c512, 1),
            conv_dw(c512, c512, 1),
            conv_dw(c512, c512, 1),
            conv_dw(c512, c1024, 2),
            conv_dw(c1024, c1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = TritonLinear(c1024, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
