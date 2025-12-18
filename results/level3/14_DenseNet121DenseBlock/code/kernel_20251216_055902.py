import torch
import torch.nn as nn
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
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["H", "W", "C_in", "C_out"],
)
@triton.jit
def conv2d_3x3_implicit_gemm_kernel(
    x_ptr,  # *f32, [N, C_in, H, W]
    w_ptr,  # *f32, [C_in, 3, 3, C_out] (contiguous)
    y_ptr,  # *f32, [N, C_out, H, W]
    N: tl.int32,
    C_in: tl.int32,
    H: tl.int32,
    W: tl.int32,
    C_out: tl.int32,
    M: tl.int32,  # N * H * W
    stride_xn: tl.int32,
    stride_xc: tl.int32,
    stride_xh: tl.int32,
    stride_xw: tl.int32,
    stride_wc: tl.int32,
    stride_wkh: tl.int32,
    stride_wkw: tl.int32,
    stride_wo: tl.int32,
    stride_yn: tl.int32,
    stride_yc: tl.int32,
    stride_yh: tl.int32,
    stride_yw: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    High-performance implicit-GEMM 3x3 conv:

      - Treat output as Y[M, C_out], where M = N*H*W
      - Accumulate over K = C_in * 3 * 3 via:
            for ci_chunk in chunks of BLOCK_K:
              for kh in 0..2:
                for kw in 0..2:
                    A[M_tile, ci_chunk] x B[ci_chunk, C_out_tile]
      - Input:  X[N, C_in, H, W]
      - Weights: W[C_in, 3, 3, C_out]  (pre-permuted & contiguous)
      - Output: Y[N, C_out, H, W]

    Memory / fusion constraints:
      - Multiple tl.load() from inputs/weights are allowed.
      - NO intermediate tl.store(): only a single final tl.store() to y_ptr.
      - All intermediates (im2col tiles, partial sums) stay in registers.
    """

    # -----------------------------
    # Tile coordinates (M, C_out)
    # -----------------------------
    pid_m = tl.program_id(axis=0)  # along flattened M = N*H*W
    pid_n = tl.program_id(axis=1)  # along output channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode (n, h, w) from flattened index M = N * H * W
    HW = H * W
    n_idx = offs_m // HW
    rem_hw = offs_m - n_idx * HW
    h_idx = rem_hw // W
    w_idx = rem_hw - h_idx * W

    # -----------------------------
    # Accumulator tile in registers
    # -----------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -----------------------------
    # Loop over input channels (C_in) in chunks of BLOCK_K,
    # and unroll 3x3 kernel positions (kh, kw).
    # No intermediate stores.
    # -----------------------------
    c_base = 0
    while c_base < C_in:
        offs_c = c_base + tl.arange(0, BLOCK_K)  # [BK] input channels chunk
        mask_c = offs_c < C_in

        # For each kernel position (kh, kw) in 3x3
        # These loops are statically unrolled by Triton.
        for kh in range(3):
            # input h-coordinate with padding=1 -> (h + kh - 1)
            h_in = h_idx[:, None] + (kh - 1)
            in_h_ok = (h_in >= 0) & (h_in < H)

            for kw in range(3):
                # input w-coordinate with padding=1 -> (w + kw - 1)
                w_in = w_idx[:, None] + (kw - 1)
                in_w_ok = (w_in >= 0) & (w_in < W)

                in_bounds = in_h_ok & in_w_ok  # [BM,1] broadcast

                # -----------------------------
                # Load A tile: X[n, ci, h_in, w_in]
                # -----------------------------
                x_ptrs = (
                    x_ptr
                    + n_idx[:, None] * stride_xn
                    + offs_c[None, :] * stride_xc
                    + h_in * stride_xh
                    + w_in * stride_xw
                )
                mask_a = mask_m[:, None] & mask_c[None, :] & in_bounds
                a = tl.load(x_ptrs, mask=mask_a, other=0.0)  # [BM, BK]

                # -----------------------------
                # Load B tile: W[ci, kh, kw, co]
                # -----------------------------
                w_ptrs = (
                    w_ptr
                    + offs_c[:, None] * stride_wc
                    + kh * stride_wkh
                    + kw * stride_wkw
                    + offs_n[None, :] * stride_wo
                )
                mask_b = mask_c[:, None] & mask_n[None, :]
                b = tl.load(w_ptrs, mask=mask_b, other=0.0)  # [BK, BN]

                # -----------------------------
                # Fused matmul-accumulate in registers
                # -----------------------------
                acc += tl.dot(a, b)

        c_base += BLOCK_K

    # -----------------------------
    # Single final store: Y[n, co, h, w]
    # -----------------------------
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + h_idx[:, None] * stride_yh
        + w_idx[:, None] * stride_yw
    )
    mask_y = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_y)


def conv2d_3x3_triton(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    3x3 conv2d with:
      - stride = 1
      - padding = 1
      - dilation = 1
      - groups = 1
      - bias = False

    Implemented as an implicit-GEMM Triton kernel optimized for Ada (RTX 4090):
      - Reorders weights to [C_in, 3, 3, C_out] for simple, division-free indexing
      - Tiles over output pixels (M = N*H*W) and output channels C_out
      - Loops over input channels in BLOCK_K chunks, unrolling 3x3 kernel
      - Uses only a single tl.store() for the final output; all intermediates
        (im2col tiles and partial accumulators) stay in registers.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, "Only float32 supported"
    assert x.dim() == 4 and weight.dim() == 4, "Expected 4D tensors"

    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w, "Channel mismatch between input and weight"
    assert KH == 3 and KW == 3, "Kernel size must be 3x3"
    assert weight.stride()[-1] == 1, "Conv weight must be contiguous"

    # Ensure predictable, coalesced layout for input
    x_c = x.contiguous()

    # Reorder weights to [C_in, 3, 3, C_out] for cheap indexing:
    #   original: [C_out, C_in, 3, 3]
    #   permuted: [C_in, 3, 3, C_out]
    w_4d = weight.permute(1, 2, 3, 0).contiguous()

    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)

    # Strides
    stride_xn, stride_xc, stride_xh, stride_xw = x_c.stride()
    stride_wc, stride_wkh, stride_wkw, stride_wo = w_4d.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    M = N * H * W  # flattened output positions

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_3x3_implicit_gemm_kernel[grid](
        x_c,
        w_4d,
        y,
        N,
        C_in,
        H,
        W,
        C_out,
        M,
        stride_xn,
        stride_xc,
        stride_xh,
        stride_xw,
        stride_wc,
        stride_wkh,
        stride_wkw,
        stride_wo,
        stride_yn,
        stride_yc,
        stride_yh,
        stride_yw,
    )

    return y


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        Triton-accelerated Dense block:
        Conv2d(3x3) layers are computed via a high-performance Triton kernel.
        """
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            layers.append(self._make_layer(in_features, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        BatchNorm2d -> ReLU -> Conv2d(3x3, padding=1, bias=False) -> Dropout
        Conv2d weights are later used by the Triton kernel in forward().
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        """
        :param x: (batch_size, num_input_features, height, width)
        :return: (batch_size, num_output_features, height, width)
        """
        features = [x]
        for layer in self.layers:
            bn = layer[0]
            relu = layer[1]
            conv = layer[2]
            dropout = layer[3]

            out = bn(x)
            out = relu(out)
            # Triton-accelerated 3x3 convolution (bias=False)
            out = conv2d_3x3_triton(out, conv.weight)
            if dropout.p > 0.0:
                out = torch.nn.functional.dropout(out, p=dropout.p, training=self.training)

            features.append(out)
            x = torch.cat(features, dim=1)
        return x
