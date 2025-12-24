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
    BLOCK_M: tl.constexpr,   # tile over M = N * H * W
    BLOCK_N: tl.constexpr,   # tile over C_out
    BLOCK_KC: tl.constexpr,  # tile over input channels (C_in)
):
    # -----------------------------
    # 2D launch grid over (M, C_out)
    # -----------------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    M = N * H * W
    hw = H * W

    mask_m = offs_m < M        # [BM]
    mask_n = offs_n < C_out    # [BN]

    # Decode flattened spatial index: (n, h, w) from offs_m
    n_idx = offs_m // hw       # [BM]
    rem_hw = offs_m % hw       # [BM]
    h_idx = rem_hw // W        # [BM]
    w_idx = rem_hw % W         # [BM]

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Channel-tile offsets
    offs_kc = tl.arange(0, BLOCK_KC)  # [BK]

    # Pre-broadcasted indices based on output grid (shared by all fused ops)
    n_b = n_idx[:, None]       # [BM, 1]
    h_b = h_idx[:, None]       # [BM, 1]
    w_b = w_idx[:, None]       # [BM, 1]
    o_b = offs_n[None, :]      # [1, BN]

    # -----------------------------
    # Main loop over input channels (K dimension)
    # -----------------------------
    for c0 in range(0, C_in, BLOCK_KC):
        c_idx = c0 + offs_kc          # [BK]
        c_mask = c_idx < C_in         # [BK]

        c_b_row = c_idx[None, :]      # [1, BK]
        c_b_col = c_idx[:, None]      # [BK, 1]

        # -------------------------
        # Unroll 3x3 spatial kernel
        # -------------------------
        for kh in tl.static_range(3):
            # Input H index with padding = 1
            h_in = h_b + (kh - 1)     # [BM, 1]
            h_in_bounds = (h_in >= 0) & (h_in < H)  # [BM, 1]

            for kw in tl.static_range(3):
                # Input W index with padding = 1
                w_in = w_b + (kw - 1)     # [BM, 1]
                w_in_bounds = (w_in >= 0) & (w_in < W)  # [BM, 1]

                # -------------------------
                # Load input tile X and fuse ReLU (on BN output)
                # -------------------------
                x_mask = (
                    mask_m[:, None] &      # [BM, 1]
                    c_mask[None, :] &      # [1, BK]
                    h_in_bounds &          # [BM, 1]
                    w_in_bounds            # [BM, 1]
                )                          # -> [BM, BK]

                x_ptrs = (
                    x_ptr
                    + n_b * stride_xn      # [BM, 1]
                    + c_b_row * stride_xc  # [1, BK]
                    + h_in * stride_xh     # [BM, 1]
                    + w_in * stride_xw     # [BM, 1]
                )                          # -> [BM, BK]

                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
                # Fused ReLU with identical indexing & mask as load
                x_vals = tl.maximum(x_vals, 0.0)
                x_vals = x_vals.to(tl.float32)

                # -------------------------
                # Load weight tile W
                # -------------------------
                w_mask = (
                    c_mask[:, None] &   # [BK, 1]
                    mask_n[None, :]     # [1, BN]
                )                       # -> [BK, BN]

                w_ptrs = (
                    w_ptr
                    + o_b * stride_wo        # [1, BN]
                    + c_b_col * stride_wc    # [BK, 1]
                    + kh * stride_wh
                    + kw * stride_ww
                )                            # -> [BK, BN]

                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
                w_vals = w_vals.to(tl.float32)

                # -------------------------
                # Accumulate GEMM
                # -------------------------
                acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # -----------------------------
    # Store result Y: [N, C_out, H, W]
    # -----------------------------
    y_ptrs = (
        y_ptr
        + n_b * stride_yn
        + o_b * stride_yc
        + h_b * stride_yh
        + w_b * stride_yw
    )  # [BM, BN]

    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


def conv3x3_relu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    High-performance 3x3 Conv2d (padding=1, stride=1, bias=False) fused with ReLU on input (BN output).

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

    y = torch.empty((N, C_out, H, W), device=x.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_KC = 32

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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_KC=BLOCK_KC,
        num_warps=4,
        num_stages=3,
    )

    # Match original input dtype
    if y.dtype != x.dtype:
        y = y.to(x.dtype)
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
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BatchNorm in PyTorch for correctness
        x = self.bn(x)
        # Fused ReLU (on BN output) + Conv2d(3x3) in Triton
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
