import torch, torch.nn as nn, triton, triton.language as tl


# ---------------------------
# Triton kernels
# ---------------------------

@triton.jit
def grouped_pointwise_conv1x1_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H, W,
    C_out, G,
    C_in_g, C_out_g,
    BLOCK_M: tl.constexpr,  # rows: NHW
    BLOCK_N: tl.constexpr,  # cols: C_out_per_group
    BLOCK_K: tl.constexpr,  # reduction: C_in_per_group
):
    # Program IDs
    pid_m = tl.program_id(0)  # over NHW
    pid_n = tl.program_id(1)  # over C_out_g
    pid_g = tl.program_id(2)  # over groups

    HW = H * W
    M = N * HW
    K = C_in_g

    # Row (NHW) and col (C_out_g) indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out_g

    # Map m -> (n, h, w)
    bn = offs_m // HW
    rem = offs_m % HW
    h = rem // W
    w = rem % W

    # Base channel index for this group
    c0 = pid_g * C_in_g
    # Base output channel index for this group
    oc = pid_g * C_out_g + offs_n  # (BLOCK_N,)

    # Precompute base offsets
    CinHW = C_in * HW
    CoutHW = C_out * HW

    base_x = bn * CinHW + h * W + w + c0 * HW        # (BLOCK_M,)
    base_y = bn * CoutHW + h * W + w                 # (BLOCK_M,)

    # Reduction indices
    offs_k = tl.arange(0, BLOCK_K)  # (BLOCK_K,)

    # Pointers for A: shape (BM, BK)
    a_ptrs = x_ptr + base_x[:, None] + offs_k[None, :] * HW
    # Pointers for B: shape (BK, BN)
    b_ptrs = w_ptr + oc[None, :] * C_in_g + offs_k[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        a_mask = mask_m[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        a = tl.astype(a, tl.float32)
        b = tl.astype(b, tl.float32)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * HW
        b_ptrs += BLOCK_K

    # Store
    y_ptrs = y_ptr + base_y[:, None] + oc[None, :] * HW
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C, H, W,
    BLOCK_P: tl.constexpr,
):
    # Each program handles one (n, c) pair over a tile of spatial positions
    pid_nc = tl.program_id(0)  # 0 .. N*C-1
    pid_p = tl.program_id(1)   # spatial tiles

    NC = N * C
    P = H * W

    # We set grid0 = NC, so pid_nc < NC always
    n = pid_nc // C
    c = pid_nc % C

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < P

    # Spatial indices
    h = offs_p // W
    w = offs_p % W

    # Precompute base index for (n, c)
    base = (n * C + c) * P

    # Load 3x3 weights for this channel
    w_base = c * 9
    w00 = tl.load(w_ptr + w_base + 0)
    w01 = tl.load(w_ptr + w_base + 1)
    w02 = tl.load(w_ptr + w_base + 2)
    w10 = tl.load(w_ptr + w_base + 3)
    w11 = tl.load(w_ptr + w_base + 4)
    w12 = tl.load(w_ptr + w_base + 5)
    w20 = tl.load(w_ptr + w_base + 6)
    w21 = tl.load(w_ptr + w_base + 7)
    w22 = tl.load(w_ptr + w_base + 8)

    acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

    # Common clamped coordinates
    h_m1 = h - 1
    h_p1 = h + 1
    w_m1 = w - 1
    w_p1 = w + 1

    h_m1_clamp = tl.maximum(0, h_m1)
    h_p1_clamp = tl.minimum(H - 1, h_p1)
    w_m1_clamp = tl.maximum(0, w_m1)
    w_p1_clamp = tl.minimum(W - 1, w_p1)

    # (h-1, w-1)
    mask00 = mask_p & (h > 0) & (w > 0)
    idx00 = base + h_m1_clamp * W + w_m1_clamp
    v00 = tl.load(x_ptr + idx00, mask=mask00, other=0.0)
    acc += tl.astype(v00, tl.float32) * tl.astype(w00, tl.float32)

    # (h-1, w)
    mask01 = mask_p & (h > 0)
    idx01 = base + h_m1_clamp * W + w
    v01 = tl.load(x_ptr + idx01, mask=mask01, other=0.0)
    acc += tl.astype(v01, tl.float32) * tl.astype(w01, tl.float32)

    # (h-1, w+1)
    mask02 = mask_p & (h > 0) & (w < W - 1)
    idx02 = base + h_m1_clamp * W + w_p1_clamp
    v02 = tl.load(x_ptr + idx02, mask=mask02, other=0.0)
    acc += tl.astype(v02, tl.float32) * tl.astype(w02, tl.float32)

    # (h, w-1)
    mask10 = mask_p & (w > 0)
    idx10 = base + h * W + w_m1_clamp
    v10 = tl.load(x_ptr + idx10, mask=mask10, other=0.0)
    acc += tl.astype(v10, tl.float32) * tl.astype(w10, tl.float32)

    # (h, w)
    mask11 = mask_p
    idx11 = base + h * W + w
    v11 = tl.load(x_ptr + idx11, mask=mask11, other=0.0)
    acc += tl.astype(v11, tl.float32) * tl.astype(w11, tl.float32)

    # (h, w+1)
    mask12 = mask_p & (w < W - 1)
    idx12 = base + h * W + w_p1_clamp
    v12 = tl.load(x_ptr + idx12, mask=mask12, other=0.0)
    acc += tl.astype(v12, tl.float32) * tl.astype(w12, tl.float32)

    # (h+1, w-1)
    mask20 = mask_p & (h < H - 1) & (w > 0)
    idx20 = base + h_p1_clamp * W + w_m1_clamp
    v20 = tl.load(x_ptr + idx20, mask=mask20, other=0.0)
    acc += tl.astype(v20, tl.float32) * tl.astype(w20, tl.float32)

    # (h+1, w)
    mask21 = mask_p & (h < H - 1)
    idx21 = base + h_p1_clamp * W + w
    v21 = tl.load(x_ptr + idx21, mask=mask21, other=0.0)
    acc += tl.astype(v21, tl.float32) * tl.astype(w21, tl.float32)

    # (h+1, w+1)
    mask22 = mask_p & (h < H - 1) & (w < W - 1)
    idx22 = base + h_p1_clamp * W + w_p1_clamp
    v22 = tl.load(x_ptr + idx22, mask=mask22, other=0.0)
    acc += tl.astype(v22, tl.float32) * tl.astype(w22, tl.float32)

    # Store result
    y_idx = base + offs_p
    tl.store(y_ptr + y_idx, acc, mask=mask_p)


@triton.jit
def channel_shuffle_kernel(
    x_ptr, y_ptr,
    B, C, H, W, G,
    BLOCK_M: tl.constexpr,  # over B*H*W
    BLOCK_N: tl.constexpr,  # over C
):
    P = H * W
    M = B * P

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C

    # Map m -> (b, s) where s is flattened spatial index
    b = offs_m // P
    s = offs_m % P

    co = offs_n  # output channels

    Cpg = C // G

    # Map output channel to input channel: ci = (co % G) * Cpg + (co // G)
    ci = (co % G) * Cpg + (co // G)

    # 2D broadcasting
    bC = b * C  # (BM,)
    bC_2d = bC[:, None]
    s_2d = s[:, None]
    co_2d = co[None, :]
    ci_2d = ci[None, :]

    x_idx = (bC_2d + ci_2d) * P + s_2d
    y_idx = (bC_2d + co_2d) * P + s_2d

    mask = mask_m[:, None] & mask_n[None, :]

    vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    tl.store(y_ptr + y_idx, vals, mask=mask)


@triton.jit
def linear_gemm_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        a_mask = mask_m[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        a = tl.astype(a, tl.float32)
        b = tl.astype(b, tl.float32)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        bias = tl.astype(bias, tl.float32)
        acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------
# Python wrappers
# ---------------------------

def grouped_pointwise_conv1x1_triton(x: torch.Tensor, weight: torch.Tensor, groups: int) -> torch.Tensor:
    """
    High-performance 1x1 grouped convolution using GEMM in Triton.
    x: [N, C_in, H, W]
    weight: [C_out, C_in/groups, 1, 1]
    """
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    N, C_in, H, W = x.shape
    C_out, C_in_g, kH, kW = weight.shape
    assert kH == 1 and kW == 1
    assert C_in % groups == 0
    assert C_in_g == C_in // groups
    assert C_out % groups == 0
    C_out_g = C_out // groups

    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)
    w2 = weight.view(C_out, C_in_g).contiguous()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    M = N * H * W
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(C_out_g, BLOCK_N),
        groups,
    )
    grouped_pointwise_conv1x1_kernel[grid](
        x, w2, y,
        N, C_in, H, W,
        C_out, groups,
        C_in_g, C_out_g,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return y


def depthwise_conv3x3_triton(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Depthwise 3x3 convolution with stride=1, padding=1.
    x: [N, C, H, W]
    weight: [C, 1, 3, 3] or [C, 3, 3]
    """
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    N, C, H, W = x.shape

    if weight.dim() == 4:
        Cw, _, kH, kW = weight.shape
        assert Cw == C and kH == 3 and kW == 3
        w2 = weight.view(C, 3, 3).contiguous()
    else:
        Cw, kH, kW = weight.shape
        assert Cw == C and kH == 3 and kW == 3
        w2 = weight.contiguous()

    y = torch.empty_like(x)

    BLOCK_P = 256
    grid = (
        N * C,  # each program along axis-0 handles a single (n,c)
        triton.cdiv(H * W, BLOCK_P),
    )
    depthwise_conv3x3_kernel[grid](
        x, w2.view(-1), y,
        N, C, H, W,
        BLOCK_P=BLOCK_P,
    )
    return y


def channel_shuffle_triton(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    Channel shuffle in Triton.
    x: [B, C, H, W]
    """
    assert x.is_cuda
    x = x.contiguous()
    B, C, H, W = x.shape
    assert C % groups == 0

    y = torch.empty_like(x)

    BLOCK_M = 64
    BLOCK_N = 64
    P = H * W
    M = B * P

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(C, BLOCK_N),
    )
    channel_shuffle_kernel[grid](
        x, y,
        B, C, H, W, groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return y


def linear_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Linear layer y = x @ weight.T + bias
    x: [M, K]
    weight: [N, K]
    bias: [N] or None
    """
    assert x.is_cuda and weight.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K

    # B is [K, N]
    b_mat = weight.t().contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    has_bias = bias is not None
    bias_ptr = bias if has_bias else x  # dummy if not used

    linear_gemm_kernel[grid](
        x, b_mat, bias_ptr, y,
        M, N, K,
        x.stride(0), x.stride(1),
        b_mat.stride(0), b_mat.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return y


# ---------------------------
# Triton-based modules
# ---------------------------

class ChannelShuffleNew(nn.Module):
    def __init__(self, groups: int):
        super(ChannelShuffleNew, self).__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            # Fallback to PyTorch for CPU
            b, c, h, w = x.size()
            g = self.groups
            assert c % g == 0
            x = x.view(b, g, c // g, h, w).transpose(1, 2).contiguous()
            return x.view(b, c, h, w)
        return channel_shuffle_triton(x, self.groups)


class ShuffleNetUnitNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitNew, self).__init__()

        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        # We still use nn.Conv2d to hold weights & init, but we do NOT call them in forward.
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1,
                               padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1,
                               padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1,
                               padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shuffle = ChannelShuffleNew(groups)

        if in_channels == out_channels:
            self.use_shortcut = False
            self.shortcut = nn.Sequential()
        else:
            self.use_shortcut = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First 1x1 grouped conv
        if x.is_cuda:
            out = grouped_pointwise_conv1x1_triton(x, self.conv1.weight, self.conv1.groups)
        else:
            out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        # Depthwise 3x3 conv
        if out.is_cuda:
            out = depthwise_conv3x3_triton(out, self.conv2.weight)
        else:
            out = self.conv2(out)
        out = self.bn2(out)

        # Channel shuffle
        out = self.shuffle(out)

        # Second 1x1 grouped conv
        if out.is_cuda:
            out = grouped_pointwise_conv1x1_triton(out, self.conv3.weight, self.conv3.groups)
        else:
            out = self.conv3(out)
        out = self.bn3(out)
        out = torch.relu(out)

        # Shortcut
        if self.use_shortcut:
            if x.is_cuda:
                residual = grouped_pointwise_conv1x1_triton(
                    x, self.shortcut[0].weight, self.shortcut[0].groups
                )
            else:
                residual = self.shortcut[0](x)
            residual = self.shortcut[1](residual)
        else:
            residual = x

        out = out + residual
        return out


class ModelNew(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        groups=3,
        stages_repeats=[3, 7, 3],
        stages_out_channels=[24, 240, 480, 960],
    ):
        super(ModelNew, self).__init__()

        # Stem conv (3x3, stride2) – kept as PyTorch Conv2d; single layer, relatively low cost
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1],
                                       stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2],
                                       stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3],
                                       stages_repeats[2], groups)

        # Final 1x1 conv – will be run via Triton 1x1 kernel in forward
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = [ShuffleNetUnitNew(in_channels, out_channels, groups)]
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnitNew(out_channels, out_channels, groups))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        # Stages
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Final 1x1 conv via Triton when on GPU
        if x.is_cuda:
            x = grouped_pointwise_conv1x1_triton(x, self.conv5.weight, groups=1)
        else:
            x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)

        # Global average pool
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # FC via Triton GEMM when on GPU
        if x.is_cuda:
            x = linear_triton(x, self.fc.weight, self.fc.bias)
        else:
            x = self.fc(x)
        return x
