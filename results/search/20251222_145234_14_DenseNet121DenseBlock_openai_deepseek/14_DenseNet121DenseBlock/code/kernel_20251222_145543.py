import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H, W, C_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wk, stride_wn,
    stride_yn, stride_yc, stride_yh, stride_yw,
    K,  # = C_in * 3 * 3
    BLOCK_M: tl.constexpr,  # tile of output positions (N*H*W)
    BLOCK_N: tl.constexpr,  # tile of output channels (C_out)
    BLOCK_K: tl.constexpr,  # tile of reduction dim (K)
):
    # Program IDs for tiling over output positions (M) and channels (N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Logical matrix sizes
    M = N * H * W

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Map linear index m -> (n_idx, h_idx, w_idx)
    hw = H * W
    n_idx = offs_m // hw
    rem = offs_m % hw
    h_idx = rem // W
    w_idx = rem % W

    # Make them 2D for broadcasting: (BLOCK_M, 1)
    n_idx = n_idx[:, None]
    h_idx = h_idx[:, None]
    w_idx = w_idx[:, None]

    # Broadcasted output channel indices: (1, BLOCK_N)
    offs_n_b = offs_n[None, :]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k_base = tl.arange(0, BLOCK_K)

    # Loop over reduction dimension K = C_in * 3 * 3
    for k_start in range(0, K, BLOCK_K):
        k_ids = k_start + offs_k_base  # (BLOCK_K,)
        mask_k = k_ids < K

        # Map k_idx -> (ci, kh, kw) where kh,kw in [0,2]
        ci = k_ids // (3 * 3)
        remk = k_ids % (3 * 3)
        kh = remk // 3
        kw = remk % 3

        # Shapes: (1, BLOCK_K) for broadcasting
        ci = ci[None, :]
        kh = kh[None, :]
        kw = kw[None, :]

        # Input spatial positions with padding=1
        h_in = h_idx + kh - 1  # (BLOCK_M, BLOCK_K)
        w_in = w_idx + kw - 1  # (BLOCK_M, BLOCK_K)

        # Broadcast batch index and channel
        n_b = n_idx               # (BLOCK_M, 1)
        ci_b = ci                 # (1, BLOCK_K)

        # Bounds for input indices
        mask_h = (h_in >= 0) & (h_in < H)
        mask_w = (w_in >= 0) & (w_in < W)
        mask_in = mask_h & mask_w

        # Full mask for A (input patches)
        mask_a = mask_in & mask_m[:, None] & mask_k[None, :]

        # Pointer arithmetic for A matrix (input patches)
        x_ptrs = (
            x_ptr
            + n_b * stride_xn
            + ci_b * stride_xc
            + h_in * stride_xh
            + w_in * stride_xw
        )

        a = tl.load(x_ptrs, mask=mask_a, other=0.0)

        # Pointer arithmetic for B matrix (weights) with shape [K, C_out]
        w_ptrs = (
            w_ptr
            + k_ids[:, None] * stride_wk
            + offs_n_b * stride_wn
        )
        mask_b = mask_k[:, None] & mask_n[None, :]

        b = tl.load(w_ptrs, mask=mask_b, other=0.0)

        # FMA via dot-product
        acc += tl.dot(a, b, allow_tf32=True)

    # Write back to NCHW output tensor
    y_ptrs = (
        y_ptr
        + n_idx * stride_yn
        + offs_n_b * stride_yc
        + h_idx * stride_yh
        + w_idx * stride_yw
    )
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_out)


def conv3x3_triton(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    High-performance 3x3 Conv2d (stride=1, padding=1, dilation=1, bias=False) for NCHW tensors.

    x:      [N, C_in, H, W]
    weight: [C_out, C_in, 3, 3]
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dim() == 4, "Input must be NCHW"
    assert weight.dim() == 4 and weight.shape[2] == 3 and weight.shape[3] == 3, "Weight must be [C_out, C_in, 3, 3]"

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]

    # Output tensor
    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)

    # Flatten weights to [K, C_out] where K = C_in * 3 * 3
    K = C_in * 3 * 3
    w_2d = weight.view(C_out, K).transpose(0, 1).contiguous()

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()
    stride_wk, stride_wn = w_2d.stride()

    M = N * H * W

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(C_out, META["BLOCK_N"]),
        )

    conv3x3_nchw_kernel[grid](
        x, w_2d, y,
        N, C_in, H, W, C_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wk, stride_wn,
        stride_yn, stride_yc, stride_yh, stride_yw,
        K,
        BLOCK_M=64,
        BLOCK_N=32,
        BLOCK_K=32,
    )
    return y


class TritonConv3x3(nn.Module):
    """
    Drop-in replacement for nn.Conv2d with:
      - kernel_size=3, stride=1, padding=1, dilation=1, bias=False
    implemented using a high-performance Triton kernel.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Match typical Conv2d init: Kaiming normal for ReLU
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv3x3_triton(x, self.weight)


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        Dense block with Triton-accelerated 3x3 Conv2d per layer.
        Layout per layer:
          BatchNorm2d -> ReLU(inplace=True) -> TritonConv3x3 -> Dropout(0.0)
        """
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            layers.append(self._make_layer(in_features, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            TritonConv3x3(in_features, growth_rate),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x
