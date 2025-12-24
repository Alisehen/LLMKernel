import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3x3_relu_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H, W, C_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # over N*H*W
    BLOCK_N: tl.constexpr,  # over C_out
    BLOCK_K: tl.constexpr,  # over C_in * 9
):
    # Program ids for 2D launch grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets in flattened M = N * H * W and N-dim (output channels)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M = N * H * W
    hw = H * W

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode flattened spatial index: (n, h, w) from offs_m
    n_idx = offs_m // hw
    rem_hw = offs_m % hw
    h_idx = rem_hw // W
    w_idx = rem_hw % W

    # Accumulator in FP32 for better numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K = C_in * 9
    offs_k_vec = tl.arange(0, BLOCK_K)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + offs_k_vec
        k_mask = offs_k < K

        # Map K index -> (c, kh, kw)
        c_idx = offs_k // 9
        rem = offs_k % 9
        kh = rem // 3 - 1  # -1, 0, 1
        kw = rem % 3 - 1   # -1, 0, 1

        # Compute input spatial coords for this (h, w, kh, kw)
        h_in = h_idx[:, None] + kh[None, :]
        w_in = w_idx[:, None] + kw[None, :]

        n_b = n_idx[:, None]
        c_b = c_idx[None, :]

        # Bounds for input (with padding = 1)
        in_bounds = (
            (n_b < N) &
            (c_b < C_in) &
            (h_in >= 0) & (h_in < H) &
            (w_in >= 0) & (w_in < W) &
            mask_m[:, None] &
            k_mask[None, :]
        )

        # Pointers into input x: [N, C_in, H, W]
        x_ptrs = (
            x_ptr +
            n_b * stride_xn +
            c_b * stride_xc +
            h_in * stride_xh +
            w_in * stride_xw
        )
        x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)
        # Fused ReLU on BN output
        x_vals = tl.maximum(x_vals, 0.0)

        # Pointers into weights w: [C_out, C_in, 3, 3]
        kh_w = kh + 1  # map -1..1 -> 0..2
        kw_w = kw + 1
        c_w = c_idx

        kh_b = kh_w[:, None]
        kw_b = kw_w[:, None]
        c_w_b = c_w[:, None]
        offs_n_b = offs_n[None, :]

        w_in_bounds = (
            (c_w_b < C_in) &
            (offs_n_b < C_out) &
            k_mask[:, None] &
            mask_n[None, :]
        )

        w_ptrs = (
            w_ptr +
            offs_n_b * stride_wo +
            c_w_b * stride_wc +
            kh_b * stride_wh +
            kw_b * stride_ww
        )
        w_vals = tl.load(w_ptrs, mask=w_in_bounds, other=0.0)

        acc += tl.dot(x_vals.to(tl.float32), w_vals.to(tl.float32), allow_tf32=True)

    # Store result to y: [N, C_out, H, W]
    y_ptrs = (
        y_ptr +
        n_idx[:, None] * stride_yn +
        offs_n[None, :] * stride_yc +
        h_idx[:, None] * stride_yh +
        w_idx[:, None] * stride_yw
    )
    out_bounds = mask_m[:, None] & (offs_n[None, :] < C_out)
    tl.store(y_ptrs, acc, mask=out_bounds)


def conv3x3_relu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    High-performance 3x3 Conv2d (padding=1, stride=1, bias=False) fused with ReLU on input.

    Args:
        x:      (N, C_in, H, W) tensor, typically output of BatchNorm2d
        weight: (C_out, C_in, 3, 3) convolution weights
    Returns:
        y:      (N, C_out, H, W)
    """
    assert x.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors"
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    assert weight.shape[1] == C_in and weight.shape[2] == 3 and weight.shape[3] == 3

    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(N * H * W, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv3x3_relu_kernel[grid](
        x, weight, y,
        N, C_in, H, W, C_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=64,
        BLOCK_N=32,
        BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return y


class DenseLayerNew(nn.Module):
    """
    Single dense layer with:
    BatchNorm2d -> ReLU (inside Triton conv) -> Conv2d(3x3, padding=1, bias=False) -> Dropout(0.0)
    """
    def __init__(self, in_features: int, growth_rate: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.weight = nn.Parameter(torch.empty(growth_rate, in_features, 3, 3))
        # Match Conv2d default initialization approximately
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BatchNorm in PyTorch for correctness (handles running stats / training vs eval)
        x = self.bn(x)
        # Fused ReLU + Conv2d(3x3) in Triton
        out = conv3x3_relu(x, self.weight)
        # Dropout with p=0.0 (no-op but kept for structural parity)
        out = self.dropout(out)
        return out


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        Dense block with Triton-accelerated 3x3 convolutions.

        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            layers.append(DenseLayerNew(in_features, growth_rate))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, dim=1)
        return x
