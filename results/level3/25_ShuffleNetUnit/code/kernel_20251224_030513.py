import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Optional helper activations (manual implementations of missing Triton funcs)
# ---------------------------------------------------------------------------

@triton.jit
def tl_sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def tl_tanh(x):
    e2x = tl.exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


@triton.jit
def tl_silu(x):
    return x * tl_sigmoid(x)


@triton.jit
def tl_gelu(x):
    # Approximate GELU (tanh version)
    k = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715
    x3 = x * x * x
    inner = k * (x + c * x3)
    return 0.5 * x * (1.0 + tl_tanh(inner))


@triton.jit
def tl_mish(x):
    return x * tl_tanh(tl.log(1.0 + tl.exp(x)))


# ---------------------------------------------------------------------------
# 1x1 Group Convolution + BatchNorm (+ optional ReLU)
# ---------------------------------------------------------------------------

@triton.jit
def conv1x1_group_bn_relu_kernel(
    x_ptr, w_ptr,
    running_mean_ptr, running_var_ptr,
    bn_weight_ptr, bn_bias_ptr,
    y_ptr,
    B, H, W,
    Cin_g, Cout_g,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    eps,
    RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over B*H*W
    BLOCK_N: tl.constexpr,  # tile over Cout per group
    BLOCK_K: tl.constexpr,  # tile over Cin per group
):
    # program ids:
    #  - pid_m: tile along B*H*W (output spatial positions)
    #  - pid_n: tile along Cout per group
    #  - pid_g: group index
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    BHW = B * H * W
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < BHW

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < Cout_g

    # Combined output mask shared by all fused ops (conv -> BN -> ReLU -> store)
    out_mask = mask_m[:, None] & mask_n[None, :]

    # Map linear BHW index -> (b, h, w)
    HW = H * W
    b = offs_m // HW
    rem = offs_m % HW
    h_idx = rem // W
    w_idx = rem % W

    # Base input pointer for this (b, h, w, group) tile
    # x layout: [B, Cin, H, W] (NCHW)
    # group input offset = pid_g * Cin_g
    gCin = pid_g * Cin_g
    x_base = (
        b * stride_xn
        + h_idx * stride_xh
        + w_idx * stride_xw
        + gCin * stride_xc
    )  # [BLOCK_M]

    # Accumulator for GEMM (float32 for precision)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over Cin_g in K tiles
    for k in range(0, Cin_g, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Cin_g

        # A: input tile [BLOCK_M, BLOCK_K]
        a_ptrs = x_ptr + x_base[:, None] + offs_k[None, :] * stride_xc
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B: weight tile [BLOCK_K, BLOCK_N]
        # w layout: [groups, Cin_g, Cout_g] contiguous -> flatten per group as [Cin_g, Cout_g]
        w_ptrs = (
            w_ptr
            + (gCin + offs_k)[:, None] * Cout_g
            + offs_n[None, :]
        )
        b_mask = mask_k[:, None] & mask_n[None, :]
        b_mat = tl.load(w_ptrs, mask=b_mask, other=0.0)

        # GEMM
        acc += tl.dot(a, b_mat, allow_tf32=True)

    # ---------------- BatchNorm + (optional) ReLU on tile -------------------
    # BN is per *global* output channel oc = pid_g * Cout_g + offs_n
    gCout = pid_g * Cout_g
    oc = gCout + offs_n  # [BLOCK_N]

    mean = tl.load(running_mean_ptr + oc, mask=mask_n, other=0.0)
    var = tl.load(running_var_ptr + oc, mask=mask_n, other=0.0)
    gamma = tl.load(bn_weight_ptr + oc, mask=mask_n, other=1.0)
    beta = tl.load(bn_bias_ptr + oc, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv_std

    # Broadcast BN params across BLOCK_M dimension, using *same* offsets & mask
    acc = (acc - mean[None, :]) * scale[None, :] + beta[None, :]

    if RELU:
        acc = tl.maximum(acc, 0.0)

    # ---------------- Store output -------------------
    # y layout: [B, Cout, H, W] (NCHW)
    y_base = (
        b * stride_yn
        + h_idx * stride_yh
        + w_idx * stride_yw
        + (gCout) * stride_yc
    )  # [BLOCK_M]
    y_ptrs = y_ptr + y_base[:, None] + offs_n[None, :] * stride_yc

    tl.store(y_ptrs, acc, mask=out_mask)


def conv1x1_group_bn_relu_triton(x: torch.Tensor,
                                 conv: nn.Conv2d,
                                 bn: nn.BatchNorm2d,
                                 relu: bool = True) -> torch.Tensor:
    """
    Fused 1x1 group convolution + BatchNorm (+ optional ReLU)
    on NCHW input/output.
    """
    assert x.is_cuda
    assert x.dim() == 4
    B, Cin, H, W = x.shape
    weight = conv.weight  # [Cout, Cin_g, 1, 1]
    Cout, Cin_g, kh, kw = weight.shape
    assert kh == 1 and kw == 1
    groups = conv.groups
    assert Cin % groups == 0
    assert Cout % groups == 0
    Cout_g = Cout // groups

    # Weight layout: [groups, Cin_g, Cout_g] contiguous (group-local GEMM)
    w_mat = weight.view(groups, Cout_g, Cin_g).permute(0, 2, 1).contiguous()

    running_mean = bn.running_mean.contiguous()
    running_var = bn.running_var.contiguous()
    if bn.weight is not None:
        bn_weight = bn.weight.contiguous()
    else:
        bn_weight = torch.ones_like(running_mean)
    if bn.bias is not None:
        bn_bias = bn.bias.contiguous()
    else:
        bn_bias = torch.zeros_like(running_mean)

    y = torch.empty((B, Cout, H, W), device=x.device, dtype=x.dtype)

    # Tile sizes chosen as powers of two
    BLOCK_M = 64   # B*H*W tile
    BLOCK_N = 64   # Cout_g tile
    BLOCK_K = 32   # Cin_g tile

    grid = (
        triton.cdiv(B * H * W, BLOCK_M),  # pid_m
        triton.cdiv(Cout_g, BLOCK_N),     # pid_n
        groups,                           # pid_g
    )

    conv1x1_group_bn_relu_kernel[grid](
        x, w_mat,
        running_mean, running_var,
        bn_weight, bn_bias,
        y,
        B, H, W,
        Cin_g, Cout_g,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        bn.eps,
        RELU=relu,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return y


# ---------------------------------------------------------------------------
# Depthwise 3x3 Convolution + BatchNorm (groups = channels)
# ---------------------------------------------------------------------------

@triton.jit
def depthwise_conv3x3_bn_kernel(
    x_ptr, w_ptr,
    running_mean_ptr, running_var_ptr,
    bn_weight_ptr, bn_bias_ptr,
    y_ptr,
    B, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    eps,
    BLOCK_BC: tl.constexpr,  # tile over B*C
    BLOCK_HW: tl.constexpr,  # tile over H*W
):
    # Grid:
    #  pid_bc over B*C  (batch-channel)
    #  pid_hw over H*W  (spatial)
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    BC = B * C
    offs_bc = pid_bc * BLOCK_BC + tl.arange(0, BLOCK_BC)
    mask_bc = offs_bc < BC

    HW = H * W
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    # Map offs_bc -> (b, c)
    b = offs_bc // C
    c = offs_bc % C

    # Map offs_hw -> (h, w)
    h_idx = offs_hw // W
    w_idx = offs_hw % W

    # Output mask for this tile, shared by conv, BN, store
    out_mask = mask_bc[:, None] & mask_hw[None, :]

    # Base offset for center pixel x[b, c, h, w] (without base pointer)
    base_offset_bc = (b * stride_xn + c * stride_xc)[:, None]          # [BLOCK_BC, 1]
    base_offset_hw = (h_idx * stride_xh + w_idx * stride_xw)[None, :]  # [1, BLOCK_HW]
    x_center = base_offset_bc + base_offset_hw                         # [BLOCK_BC, BLOCK_HW]

    # Accumulator for depthwise conv
    acc = tl.zeros((BLOCK_BC, BLOCK_HW), dtype=tl.float32)

    # Pre-load BN scalars per channel (1D over c)
    mean = tl.load(running_mean_ptr + c, mask=mask_bc, other=0.0)
    var = tl.load(running_var_ptr + c, mask=mask_bc, other=0.0)
    gamma = tl.load(bn_weight_ptr + c, mask=mask_bc, other=1.0)
    beta = tl.load(bn_bias_ptr + c, mask=mask_bc, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = gamma * inv_std

    # Depthwise 3x3 with padding=1, stride=1
    # All ops share (offs_bc, offs_hw, out_mask) as their output indexing.
    for dh in range(-1, 2):
        ih = h_idx[None, :] + dh
        mask_h = (ih >= 0) & (ih < H)

        # Scalar offset for row
        row_offset = dh * stride_xh

        for dw in range(-1, 2):
            iw = w_idx[None, :] + dw
            mask_w = (iw >= 0) & (iw < W)

            m = out_mask & mask_h & mask_w

            col_offset = dw * stride_xw

            # IMPORTANT FIX: include base pointer so x_ptrs is a proper pointer tensor
            x_ptrs = x_ptr + x_center + row_offset + col_offset

            x_val = tl.load(x_ptrs, mask=m, other=0.0)

            # Weight index for this (dh, dw)
            k_index = (dh + 1) * 3 + (dw + 1)
            w_ptrs = w_ptr + c * 9 + k_index
            # Broadcast weights along HW dimension
            w_val = tl.load(w_ptrs, mask=mask_bc, other=0.0)
            acc += w_val[:, None] * x_val

    # Apply BatchNorm using same (b,c,h,w) offsets and out_mask
    acc = (acc - mean[:, None]) * scale[:, None] + beta[:, None]

    # Store y[b, c, h, w]
    y_ptrs = (
        y_ptr
        + (b * stride_yn + c * stride_yc)[:, None]
        + (h_idx * stride_yh + w_idx * stride_yw)[None, :]
    )
    tl.store(y_ptrs, acc, mask=out_mask)


def depthwise_conv3x3_bn_triton(x: torch.Tensor,
                                conv: nn.Conv2d,
                                bn: nn.BatchNorm2d) -> torch.Tensor:
    """
    Fused depthwise 3x3 convolution + BatchNorm (stride=1, padding=1).
    Depthwise: groups = channels = out_channels.
    Grid covers output [B, C, H, W] with a 2D tiling on BC and HW.
    """
    assert x.is_cuda
    assert x.dim() == 4
    B, C, H, W = x.shape
    weight = conv.weight  # [C, 1, 3, 3]
    assert conv.groups == C
    assert weight.shape[0] == C and weight.shape[2] == 3 and weight.shape[3] == 3

    # Compact [C, 3, 3] layout
    w_mat = weight.view(C, 3, 3).contiguous()

    running_mean = bn.running_mean.contiguous()
    running_var = bn.running_var.contiguous()
    if bn.weight is not None:
        bn_weight = bn.weight.contiguous()
    else:
        bn_weight = torch.ones_like(running_mean)
    if bn.bias is not None:
        bn_bias = bn.bias.contiguous()
    else:
        bn_bias = torch.zeros_like(running_mean)

    y = torch.empty_like(x)

    # Tile sizes; powers of two
    BLOCK_BC = 32
    BLOCK_HW = 64

    grid = (
        triton.cdiv(B * C, BLOCK_BC),  # pid_bc
        triton.cdiv(H * W, BLOCK_HW),  # pid_hw
    )

    depthwise_conv3x3_bn_kernel[grid](
        x, w_mat,
        running_mean, running_var,
        bn_weight, bn_bias,
        y,
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        bn.eps,
        BLOCK_BC=BLOCK_BC, BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2,
    )
    return y


# ---------------------------------------------------------------------------
# Channel Shuffle
# ---------------------------------------------------------------------------

@triton.jit
def channel_shuffle_kernel(
    x_ptr, y_ptr,
    B, C, H, W,
    groups,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tile over B*H*W
    BLOCK_C: tl.constexpr,  # tile over C
):
    # Grid:
    #  pid_m over B*H*W
    #  pid_c over C
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    BHW = B * H * W
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < BHW

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Joint mask and offsets for fused read/write of same output region
    mask = mask_m[:, None] & mask_c[None, :]

    # Map offs_m -> (b, h, w)
    HW = H * W
    b = offs_m // HW
    rem = offs_m % HW
    h_idx = rem // W
    w_idx = rem % W

    # Channel shuffle index mapping
    co = offs_c  # output channel index
    Cg = C // groups
    ci = (co % groups) * Cg + (co // groups)  # input channel index after shuffle

    # Input: x[b, ci, h, w]
    x_ptrs = (
        x_ptr
        + b[:, None] * stride_xn
        + ci[None, :] * stride_xc
        + h_idx[:, None] * stride_xh
        + w_idx[:, None] * stride_xw
    )
    val = tl.load(x_ptrs, mask=mask, other=0.0)

    # Output: y[b, co, h, w]
    y_ptrs = (
        y_ptr
        + b[:, None] * stride_yn
        + co[None, :] * stride_yc
        + h_idx[:, None] * stride_yh
        + w_idx[:, None] * stride_yw
    )
    tl.store(y_ptrs, val, mask=mask)


def channel_shuffle_triton(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel shuffle: x [B, C, H, W], C divisible by groups.
    Uses 2D grid over (B*H*W, C); loads and stores share identical
    (offs_m, offs_c, mask).
    """
    assert x.is_cuda
    B, C, H, W = x.shape
    assert C % groups == 0

    y = torch.empty_like(x)

    BLOCK_M = 64
    BLOCK_C = 64

    grid = (
        triton.cdiv(B * H * W, BLOCK_M),
        triton.cdiv(C, BLOCK_C),
    )

    channel_shuffle_kernel[grid](
        x, y,
        B, C, H, W,
        groups,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=1,
    )
    return y


class ChannelShuffleTriton(nn.Module):
    def __init__(self, groups: int):
        super(ChannelShuffleTriton, self).__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return channel_shuffle_triton(x, self.groups)


# ---------------------------------------------------------------------------
# Fused Add + ReLU
# ---------------------------------------------------------------------------

@triton.jit
def add_relu_kernel(
    a_ptr, b_ptr, out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    # 1D grid over all elements, offsets shared by add and ReLU
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    c = a + b
    c = tl.maximum(c, 0.0)
    tl.store(out_ptr + offs, c, mask=mask)


def add_relu_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Fused elementwise add + ReLU: out = relu(a + b)
    Single 1D grid; all fused ops share the same (offs, mask).
    """
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    out = torch.empty_like(a)
    N = a.numel()
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    add_relu_kernel[grid](a, b, out, N, BLOCK=BLOCK, num_warps=4, num_stages=1)
    return out


# ---------------------------------------------------------------------------
# ModelNew: ShuffleNet Unit with Triton Kernels
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit using Triton-fused kernels:
        - 1x1 group conv + BN + ReLU
        - depthwise 3x3 conv + BN
        - channel shuffle
        - 1x1 group conv + BN + ReLU
        - optional shortcut 1x1 conv + BN
        - fused add + ReLU
        """
        super(ModelNew, self).__init__()

        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        # First 1x1 group convolution
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels,
            kernel_size=1, stride=1, padding=0,
            groups=groups, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # Depthwise 3x3 convolution
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels,
            kernel_size=3, stride=1, padding=1,
            groups=mid_channels, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # Second 1x1 group convolution
        self.conv3 = nn.Conv2d(
            mid_channels, out_channels,
            kernel_size=1, stride=1, padding=0,
            groups=groups, bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shuffle operation (Triton)
        self.shuffle = ChannelShuffleTriton(groups)

        # Shortcut connection
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == out_channels:
            self.shortcut_conv = None
            self.shortcut_bn = None
        else:
            self.shortcut_conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=1, padding=0,
                bias=False,
            )
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main branch
        out = conv1x1_group_bn_relu_triton(x, self.conv1, self.bn1, relu=True)
        out = depthwise_conv3x3_bn_triton(out, self.conv2, self.bn2)
        out = self.shuffle(out)
        out = conv1x1_group_bn_relu_triton(out, self.conv3, self.bn3, relu=True)

        # Shortcut branch
        if self.shortcut_conv is None:
            residual = x
        else:
            # Shortcut: conv + BN (no ReLU)
            residual = conv1x1_group_bn_relu_triton(
                x, self.shortcut_conv, self.shortcut_bn, relu=False
            )

        # Fused add + ReLU
        out = add_relu_triton(out, residual)
        return out
