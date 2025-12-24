import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline, conservative
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_KC": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Larger M tile, more reuse along H*W, moderate warps
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_KC": 32},
            num_warps=4,
            num_stages=3,
        ),
        # Larger N tile (C_out), good when C_out large
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_KC": 32},
            num_warps=8,
            num_stages=2,
        ),
        # Aggressive: large tiles in both M and N
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_KC": 32},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["N", "C_in", "H", "W", "C_out"],
)
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
    # 2D launch grid over (M, C_out)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Flattened output index M = N * H * W
    M = N * H * W
    HW = H * W

    # Offsets within tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    # Bounds masks
    mask_m = offs_m < M          # [BM]
    mask_n = offs_n < C_out      # [BN]
    mask_m_b = mask_m[:, None]   # [BM, 1]
    mask_n_b = mask_n[None, :]   # [1, BN]

    # Decode (n, h, w) from flattened M index
    n_idx = offs_m // HW         # [BM]
    rem_hw = offs_m % HW         # [BM]
    h_idx = rem_hw // W          # [BM]
    w_idx = rem_hw % W           # [BM]

    # Broadcasted for pointer arithmetic
    n_b = n_idx[:, None]         # [BM, 1]
    h_b = h_idx[:, None]         # [BM, 1]
    w_b = w_idx[:, None]         # [BM, 1]
    o_b = offs_n[None, :]        # [1, BN]

    # Accumulator in FP32: [BM, BN]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Channel-tile offsets
    offs_kc = tl.arange(0, BLOCK_KC)  # [BK]

    # Main loop over input channels (K dimension)
    for c0 in range(0, C_in, BLOCK_KC):
        c_idx = c0 + offs_kc          # [BK]
        c_mask = c_idx < C_in         # [BK]

        c_row = c_idx[None, :]        # [1, BK]
        c_col = c_idx[:, None]        # [BK, 1]

        # Base X tile pointer (no 3x3 offsets yet)
        # X: [N, C_in, H, W]
        x_base = (
            x_ptr
            + n_b * stride_xn         # [BM, 1]
            + c_row * stride_xc       # [1, BK]
            + h_b * stride_xh         # [BM, 1]
            + w_b * stride_xw         # [BM, 1]
        )  # -> [BM, BK]

        # Base W tile pointer (no 3x3 offsets yet)
        # W: [C_out, C_in, 3, 3]
        w_base = (
            w_ptr
            + o_b * stride_wo         # [1, BN]
            + c_col * stride_wc       # [BK, 1]
        )  # -> [BK, BN]

        # Weight mask only depends on channel and C_out bounds
        w_mask = c_mask[:, None] & mask_n_b  # [BK, BN]

        # Unrolled 3x3 kernel over spatial offsets
        for kh in tl.static_range(3):
            h_off = kh - 1

            h_in = h_idx + h_off                    # [BM]
            h_in_bounds = (h_in >= 0) & (h_in < H)  # [BM]
            h_in_bounds_b = h_in_bounds[:, None]    # [BM, 1]

            x_base_h = x_base + h_off * stride_xh   # [BM, BK]
            w_base_h = w_base + kh * stride_wh      # [BK, BN]

            for kw in tl.static_range(3):
                w_off = kw - 1

                w_in = w_idx + w_off                    # [BM]
                w_in_bounds = (w_in >= 0) & (w_in < W)  # [BM]
                w_in_bounds_b = w_in_bounds[:, None]    # [BM, 1]

                # Combined mask for X load
                x_mask = (
                    mask_m_b &                  # [BM, 1]
                    c_mask[None, :] &           # [1, BK]
                    h_in_bounds_b &             # [BM, 1]
                    w_in_bounds_b               # [BM, 1]
                )  # -> [BM, BK]

                # Final X and W pointers for this (kh, kw)
                x_ptrs = x_base_h + w_off * stride_xw   # [BM, BK]
                w_ptrs = w_base_h + kw * stride_ww      # [BK, BN]

                # Load X tile, apply ReLU in-place
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
                x_vals = tl.maximum(x_vals, 0.0)

                # Load W tile
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                # Accumulate GEMM
                # For fp16/bf16 inputs this uses tensor cores with fp32 accumulate.
                acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # Store result Y: [N, C_out, H, W]
    y_ptrs = (
        y_ptr
        + n_b * stride_yn
        + o_b * stride_yc
        + h_b * stride_yh
        + w_b * stride_yw
    )  # [BM, BN]

    out_mask = mask_m_b & mask_n_b
    tl.store(y_ptrs, acc, mask=out_mask)


def conv3x3_relu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    3x3 Conv2d (padding=1, stride=1, bias=False) with ReLU applied to the input.
    x:      (N, C_in, H, W)
    weight: (C_out, C_in, 3, 3)
    returns y: (N, C_out, H, W), same dtype as x
    """
    assert x.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors"
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    assert weight.shape[1] == C_in and weight.shape[2] == 3 and weight.shape[3] == 3

    # Accumulate in fp32 internally; output buffer is fp32, then cast to x.dtype
    y = torch.empty((N, C_out, H, W), device=x.device, dtype=torch.float32)

    M = N * H * W

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(C_out, META["BLOCK_N"]),
        )

    conv3x3_relu_kernel[grid](
        x, weight, y,
        N, C_in, H, W, C_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )

    if y.dtype != x.dtype:
        y = y.to(x.dtype)
    return y


class DenseLayerNew(nn.Module):
    """
    Single dense layer:
    BatchNorm2d -> ReLU (inside Triton conv) -> Conv2d(3x3, padding=1, bias=False) -> Dropout(0.0)
    """
    def __init__(self, in_features: int, growth_rate: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.weight = nn.Parameter(torch.empty(growth_rate, in_features, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=2.23606797749979)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        out = conv3x3_relu(x, self.weight)
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
