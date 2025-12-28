# <optimized Triton code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------
# Helper elementwise functions (can be used in fused kernels)
# ---------------------------------------------------------


def _tl_tanh(x):
    # tanh(x) = 2 * sigmoid(2x) - 1
    e2x = tl.exp(-2 * x)
    return (1 - e2x) / (1 + e2x)


def _tl_sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


def _tl_gelu(x):
    # Approximate GELU (tanh formulation)
    c = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + c * x_cubed)
    return 0.5 * x * (1.0 + _tl_tanh(inner)) * x


def _tl_silu(x):
    return x * _tl_sigmoid(x)


def _tl_mish(x):
    # mish(x) = x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x))
    softplus = tl.log(1.0 + tl.exp(x))
    return x * _tl_tanh(softplus)


# ---------------------------------------------------------
# Triton Kernels
# ---------------------------------------------------------


@triton.autotune(
    configs=[
        # Aggressive config for large GEMMs on 4090
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        # Slightly narrower N for tall matrices
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
        # Balanced baseline (conservative)
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        # Small fallback for tiny M/N
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_gemm_bias_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # logical [K, N] (may be a view of [N, K])
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,  # tile in M
    BLOCK_N: tl.constexpr,  # tile in N
    BLOCK_K: tl.constexpr,  # tile in K
):
    # 2D grid over output [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_mn = mask_m[:, None] & mask_n[None, :]

    # Pointers to the first K-tile for A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k in range(0, K, BLOCK_K):
        k_rem = K - k
        k_mask = offs_k < k_rem

        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & mask_n[None, :],
            other=0.0,
        )
        # TF32 enabled for speed on 4090 (SM89+)
        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused bias add (broadcast along M) using the same N offsets
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    # Single final store (no intermediates written to memory)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_mn)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
    ],
    key=["NUMEL"],
)
@triton.jit
def add_relu_kernel(
    a_ptr,  # input A
    b_ptr,  # input B
    c_ptr,  # output
    NUMEL,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid over flat elements
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NUMEL

    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)

    # Fused add + ReLU keeps everything in registers
    out = a + b
    out = tl.maximum(out, 0.0)

    # Single final store
    tl.store(c_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_NC": 128, "BLOCK_HW": 128},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_NC": 128, "BLOCK_HW": 64},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_NC": 64, "BLOCK_HW": 64},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["C", "H", "W"],
)
@triton.jit
def global_avgpool2d_kernel(
    x_ptr,   # [N, C, H, W]
    y_ptr,   # [N, C]
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    stride_on, stride_oc,
    BLOCK_NC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # 1D grid over flattened (N * C)
    pid = tl.program_id(0)
    offs_nc = pid * BLOCK_NC + tl.arange(0, BLOCK_NC)
    nc_mask = offs_nc < (N * C)

    # Decode (n, c) from flattened index
    n = offs_nc // C
    c = offs_nc % C

    S = H * W
    acc = tl.zeros((BLOCK_NC,), dtype=tl.float32)

    # Reduction over spatial dimension
    for s in range(0, S, BLOCK_HW):
        offs_s = s + tl.arange(0, BLOCK_HW)
        s_mask = offs_s < S

        h = offs_s // W
        w = offs_s % W

        ptrs = (
            x_ptr
            + n[:, None] * stride_n
            + c[:, None] * stride_c
            + h[None, :] * stride_h
            + w[None, :] * stride_w
        )

        vals = tl.load(
            ptrs,
            mask=nc_mask[:, None] & s_mask[None, :],
            other=0.0,
        )
        # Sum over spatial dimension for each (n, c)
        acc += tl.sum(vals, axis=1)

    # Normalize by number of spatial elements
    norm = 1.0 / (H * W)
    acc = acc * norm

    # Single final store to [N, C]
    y_ptrs = y_ptr + n * stride_on + c * stride_oc
    tl.store(y_ptrs, acc, mask=nc_mask)


# ---------------------------------------------------------
# Python Wrappers
# ---------------------------------------------------------


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: [M, K]
    weight: [N, K] (standard PyTorch Linear weight layout)
    bias: [N]
    Returns: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32

    M, K = x.shape
    N, K_w = weight.shape
    assert K_w == K, "Weight shape must be [N, K] matching x[..., K]"

    # No material transpose: treat weight as logical [K, N] via strides
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            max(1, triton.cdiv(M, meta["BLOCK_M"])),
            max(1, triton.cdiv(N, meta["BLOCK_N"])),
        )

    linear_gemm_bias_kernel[grid](
        x,
        weight,  # logical [K, N]
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        # Map [N, K] -> logical [K, N] via strides
        weight.stride(1),  # stride over K dimension
        weight.stride(0),  # stride over N dimension
        y.stride(0),
        y.stride(1),
    )
    return y


def triton_add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise (a + b).relu(), fused.
    """
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    c = torch.empty_like(a)

    a_flat = a.view(-1)
    b_flat = b.view(-1)
    c_flat = c.view(-1)
    numel = a_flat.numel()

    def grid(meta):
        return (max(1, triton.cdiv(numel, meta["BLOCK_SIZE"])),)

    add_relu_kernel[grid](
        a_flat,
        b_flat,
        c_flat,
        numel,
    )
    return c


def triton_global_avgpool2d(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N, C, H, W] -> [N, C, 1, 1] (mean over H,W)
    """
    assert x.is_cuda
    assert x.ndim == 4
    N, C, H, W = x.shape

    y = torch.empty((N, C), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (max(1, triton.cdiv(N * C, meta["BLOCK_NC"])),)

    x_contig = x.contiguous()
    global_avgpool2d_kernel[grid](
        x_contig,
        y,
        N,
        C,
        H,
        W,
        x_contig.stride(0),
        x_contig.stride(1),
        x_contig.stride(2),
        x_contig.stride(3),
        y.stride(0),
        y.stride(1),
    )
    return y.view(N, C, 1, 1)


# ---------------------------------------------------------
# Conv-BN Folding Utilities
# ---------------------------------------------------------


def fold_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fold BatchNorm2d into preceding Conv2d for inference:
    y = BN(Conv(x))  ==>  y = Conv_fused(x)
    """
    assert isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)

    # Create a new conv with identical hyper-parameters but bias enabled
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    ).to(conv.weight.device, conv.weight.dtype)

    # Prepare params
    W = conv.weight
    if conv.bias is not None:
        b = conv.bias
    else:
        b = torch.zeros(conv.out_channels, device=W.device, dtype=W.dtype)

    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    # BN folding
    var_rsqrt = torch.rsqrt(running_var + eps)
    scale = gamma * var_rsqrt  # [C_out]

    W_fused = W * scale.view(-1, 1, 1, 1)
    b_fused = beta + (b - running_mean) * scale

    with torch.no_grad():
        fused_conv.weight.copy_(W_fused)
        fused_conv.bias.copy_(b_fused)
        fused_conv.weight.requires_grad_(False)
        fused_conv.bias.requires_grad_(False)

    return fused_conv


# ---------------------------------------------------------
# Model Definition
# ---------------------------------------------------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

        # Inference-only fused modules
        self.conv1_fused: nn.Conv2d | None = None
        self.conv2_fused: nn.Conv2d | None = None
        self.downsample_fused: nn.Conv2d | None = None
        self._is_fused: bool = False

    def fuse_for_inference(self):
        """
        Create Conv+BN folded versions for inference.
        """
        if self._is_fused:
            return

        # Fold main path conv+bn pairs
        self.conv1_fused = fold_conv_bn_eval(self.conv1, self.bn1)
        self.conv2_fused = fold_conv_bn_eval(self.conv2, self.bn2)

        # Fold downsample if present (Conv+BN in a Sequential)
        if self.downsample is not None:
            assert isinstance(self.downsample, nn.Sequential) and len(self.downsample) == 2
            ds_conv, ds_bn = self.downsample[0], self.downsample[1]
            self.downsample_fused = fold_conv_bn_eval(ds_conv, ds_bn)
        else:
            self.downsample_fused = None

        self._is_fused = True

    def forward(self, x):
        identity = x

        # Inference path: use fused Conv (Conv+BN folded), no BN kernels
        if self._is_fused and not self.training:
            out = self.conv1_fused(x)
            out = self.relu(out)

            out = self.conv2_fused(out)

            if self.downsample is not None:
                identity = self.downsample_fused(x)

            # Fused residual add + ReLU via Triton
            out = triton_add_relu(out, identity)
            return out

        # Training (or non-fused) path: original Conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused residual add + ReLU via Triton
        out = triton_add_relu(out, identity)

        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Keep for structure, but we'll use Triton global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Use nn.Linear for parameter management; forward uses Triton kernel
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Inference-only fused conv1 (Conv+BN folded)
        self.conv1_fused: nn.Conv2d | None = None
        self._is_fused: bool = False

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def fuse_for_inference(self):
        """
        Fold all Conv2d+BatchNorm2d pairs into single Conv2d with bias for inference.
        """
        if self._is_fused:
            return

        # Fold stem conv1 + bn1
        self.conv1_fused = fold_conv_bn_eval(self.conv1, self.bn1)

        # Fold inside all residual blocks
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.fuse_for_inference()

        self._is_fused = True

    def eval(self):
        """
        Override eval to also perform Conv+BN folding for inference.
        """
        super().eval()
        self.fuse_for_inference()
        return self

    def forward(self, x):
        # x: [B, 3, H, W]

        # Stem: use fused Conv+BN when in inference mode
        if self._is_fused and not self.training:
            x = self.conv1_fused(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Triton global average pooling instead of PyTorch AdaptiveAvgPool2d
        x = triton_global_avgpool2d(x)  # [B, 512, 1, 1]

        x = torch.flatten(x, 1)  # [B, 512]

        # Triton linear instead of nn.Linear forward
        x = triton_linear(x, self.fc.weight, self.fc.bias)

        return x
