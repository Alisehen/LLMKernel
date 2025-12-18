import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_gemm_bias_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N]
    bias_ptr,  # [N] or unused
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling of the output matrix
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows (M) and cols (N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for the first K-tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator (FP32 for good numerical stability)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k

        # Masks to avoid out-of-bounds loads
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply for this K-tile
        acc += tl.dot(a, b, allow_tf32=True)

        # Move pointers to next K-tile
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Optionally add bias
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    # Write output back
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K]  (same as nn.Linear: out_features, in_features)
    bias:   [N] or None
    returns [M, N]
    """
    assert x.is_cuda and weight.is_cuda, "Triton linear expects CUDA tensors"
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"Incompatible shapes: x ({M}, {K}) and weight ({N}, {K_w})"

    # We compute C = A @ B, with:
    #   A = x          [M, K]
    #   B = weight^T   [K, N]
    # So we materialize weight^T for coalesced access.
    b = weight.t().contiguous()  # [K, N]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    HAS_BIAS = bias is not None
    bias_ptr = bias if HAS_BIAS else x  # dummy ptr if no bias

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_gemm_bias_kernel[grid](
        x, b, bias_ptr, y,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        y.stride(0), y.stride(1),
        HAS_BIAS=HAS_BIAS,
    )
    return y


def triton_conv1x1(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    1x1 convolution via GEMM:
    x:      [B, C_in, H, W]
    weight: [C_out, C_in, 1, 1] (as in nn.Conv2d)
    returns [B, C_out, H, W]
    """
    assert x.is_cuda and weight.is_cuda, "Triton conv1x1 expects CUDA tensors"
    B, C_in, H, W = x.shape
    C_out, C_in_w, kh, kw = weight.shape
    assert kh == 1 and kw == 1, "triton_conv1x1 only supports 1x1 kernels"
    assert C_in == C_in_w, "Input channels mismatch"
    # No bias here: matches conv_final(bias=False)

    # Reshape to 2D: each spatial location is a row
    M = B * H * W
    K = C_in

    x_2d = x.permute(0, 2, 3, 1).contiguous().view(M, K)  # [M, C_in]
    w_2d = weight.view(C_out, C_in)  # [C_out, C_in] == [N, K]

    y_2d = triton_linear(x_2d, w_2d, bias=None)  # [M, C_out]

    # Back to [B, C_out, H, W]
    y = y_2d.view(B, H, W, C_out).permute(0, 3, 1, 2).contiguous()
    return y


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB2 architecture implementation with Triton-accelerated
        final 1x1 convolution and fully-connected layer.
        """
        super(ModelNew, self).__init__()

        # Initial stem
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # MBConv blocks (kept as in the original model)
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)

        # Final layers
        # Keep nn.Conv2d & nn.BatchNorm2d for state_dict compatibility;
        # use Triton for the 1x1 conv in forward().
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Keep nn.Linear for parameters; use Triton for computation.
        self.fc = nn.Linear(1408, num_classes)

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """
        Helper function to create a MBConv block.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the MBConv block.
        :return: A sequential container of layers forming the MBConv block.
        """
        layers = []
        expanded_channels = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))

        # Depthwise convolution
        layers.append(
            nn.Conv2d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=expanded_channels,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))

        # Squeeze and Excitation
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(
            nn.Conv2d(
                expanded_channels,
                expanded_channels // 4,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                expanded_channels // 4,
                expanded_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        layers.append(nn.Sigmoid())

        # Output phase
        layers.append(
            nn.Conv2d(
                expanded_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the EfficientNetB2 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))

        # MBConv blocks
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)

        # Final 1x1 conv via Triton
        x = triton_conv1x1(x, self.conv_final.weight)
        x = self.relu(self.bn_final(x))

        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Final linear via Triton (GEMM + bias)
        x = triton_linear(x, self.fc.weight, self.fc.bias)
        return x
