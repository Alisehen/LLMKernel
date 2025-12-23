# optimized Triton code

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------
# ConvTranspose3d via implicit GEMM (compute-bound, autotuned)
# ---------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["N", "C_out", "D_out", "H_out", "W_out"],
)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    stride, padding,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    w_stride_ci, w_stride_co, w_stride_kd, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    K: tl.constexpr,          # kernel size (assumed cubic)
    C_IN: tl.constexpr,       # in_channels (loop bound, compile-time)
    BLOCK_M: tl.constexpr,    # number of output positions per block
    BLOCK_N: tl.constexpr,    # number of output channels per block
    BLOCK_K: tl.constexpr,    # reduction dim tile
):
    # Implicit-GEMM formulation:
    #   A: [M, K_tot]  where  M = N * D_out * H_out * W_out,
    #                     K_tot = C_in * K^3
    #   B: [K_tot, C_out]
    #   C: [M, C_out]

    pid_m = tl.program_id(0)  # tiles over M (batch+spatial)
    pid_n = tl.program_id(1)  # tiles over output channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M = N * D_out * H_out * W_out
    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode offs_m -> (n, z_out, y_out, x_out)
    DHW_out = D_out * H_out * W_out
    HW_out = H_out * W_out

    n_idx = offs_m // DHW_out
    rem = offs_m % DHW_out
    z_idx = rem // HW_out
    rem2 = rem % HW_out
    y_idx = rem2 // W_out
    x_idx = rem2 % W_out

    # Base pointers for x and y
    y_ptr_base = (
        y_ptr
        + n_idx * y_stride_n
        + z_idx * y_stride_d
        + y_idx * y_stride_h
        + x_idx * y_stride_w
    )
    x_ptr_base = x_ptr + n_idx * x_stride_n

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Total K-dimension and kernel geometry
    RS = K * K * K                   # spatial kernel volume
    K_TOTAL = C_IN * RS              # full reduction dim
    pad_up = K - 1 - padding         # "effective padding" as in original kernel

    stride_z = stride
    stride_y = stride
    stride_x = stride

    # Decode once for output coords (for broadcasting)
    z_out = z_idx[:, None]
    y_out = y_idx[:, None]
    x_out = x_idx[:, None]

    # Loop over K-dimension tiles
    for k0 in range(0, K_TOTAL, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOTAL

        # Decode flattened K index into (ci, kd, kh, kw)
        ci = offs_k // RS
        remk = offs_k % RS
        kd = remk // (K * K)
        remk2 = remk % (K * K)
        kh = remk2 // K
        kw = remk2 % K

        # ------------------
        # Build A tile: [BLOCK_M, BLOCK_K] = im2col(x) for transposed conv
        # ------------------
        kd_b = kd[None, :]
        kh_b = kh[None, :]
        kw_b = kw[None, :]

        # Compute "upsampled" indices and map back to input coords
        iu_z = z_out + pad_up - kd_b
        iu_y = y_out + pad_up - kh_b
        iu_x = x_out + pad_up - kw_b

        iz = iu_z // stride_z
        iy = iu_y // stride_y
        ix = iu_x // stride_x

        # Validity masks per dimension
        mask_z_pos = iu_z >= 0
        mask_z_in = (iz >= 0) & (iz < D_in)
        mask_z_stride = (iu_z % stride_z) == 0

        mask_y_pos = iu_y >= 0
        mask_y_in = (iy >= 0) & (iy < H_in)
        mask_y_stride = (iu_y % stride_y) == 0

        mask_x_pos = iu_x >= 0
        mask_x_in = (ix >= 0) & (ix < W_in)
        mask_x_stride = (iu_x % stride_x) == 0

        # Combined mask for input element existence
        mask_in = (
            mask_m[:, None] & mask_k[None, :] &
            mask_z_pos & mask_z_in & mask_z_stride &
            mask_y_pos & mask_y_in & mask_y_stride &
            mask_x_pos & mask_x_in & mask_x_stride
        )

        # Broadcast ci to (BM,BK)
        ci_b = ci[None, :]

        # Compute input pointers for A tile
        x_ptrs = (
            x_ptr_base[:, None]
            + ci_b * x_stride_c
            + iz * x_stride_d
            + iy * x_stride_h
            + ix * x_stride_w
        )

        a_tile = tl.load(x_ptrs, mask=mask_in, other=0.0)

        # ------------------
        # Build B tile: [BLOCK_K, BLOCK_N] from weights
        # ------------------
        ci_col = ci[:, None]
        kd_col = kd[:, None]
        kh_col = kh[:, None]
        kw_col = kw[:, None]
        co_row = offs_n[None, :]

        w_ptrs = (
            w_ptr
            + ci_col * w_stride_ci
            + co_row * w_stride_co
            + kd_col * w_stride_kd
            + kh_col * w_stride_kh
            + kw_col * w_stride_kw
        )

        mask_w = mask_k[:, None] & mask_n[None, :]
        b_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # ------------------
        # Matmul update
        # ------------------
        acc += tl.dot(a_tile, b_tile, allow_tf32=True)

    # Add bias (epilogue)
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    # Store C tile
    y_ptrs = y_ptr_base[:, None] + offs_n[None, :] * y_stride_c
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


def conv_transpose3d_triton(x, weight, bias, kernel_size, stride, padding):
    """
    x: [N, C_in, D_in, H_in, W_in]
    weight: [C_in, C_out, K, K, K] (PyTorch ConvTranspose3d layout)
    bias: [C_out]
    """
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert bias is not None
    device = x.device

    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out, Kd, Kh, Kw = weight.shape
    assert C_in_w == C_in
    assert Kd == Kh == Kw == kernel_size

    S = stride
    P = padding
    K = kernel_size

    # Output shape (no dilation, output_padding = 0)
    D_out = (D_in - 1) * S - 2 * P + K
    H_out = (H_in - 1) * S - 2 * P + K
    W_out = (W_in - 1) * S - 2 * P + K

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=device, dtype=x.dtype)

    # Strides
    x_strides = x.stride()
    w_strides = weight.stride()
    y_strides = y.stride()

    # Flatten spatial+batch dimension
    M = N * D_out * H_out * W_out

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv_transpose3d_kernel[grid](
        x, weight, bias, y,
        N, C_in, C_out,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        S, P,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3], x_strides[4],
        w_strides[0], w_strides[1], w_strides[2], w_strides[3], w_strides[4],
        y_strides[0], y_strides[1], y_strides[2], y_strides[3], y_strides[4],
        K, C_in,
    )
    return y


# ---------------------------------------------
# Fused 3D maxpool (two stages) + sum over channels
# ---------------------------------------------
# We replace:
#   MaxPool3d(K1, S1) -> MaxPool3d(K2, S2) -> sum_channels
# with a single kernel that:
#   - uses a 1D grid over final output [N, 1, D_out, H_out, W_out]
#   - for each (n, z, y, x):
#       * computes a single max over the equivalent K_total^3 region
#         (K_total = S1*(K2-1) + K1, S_total = S1*S2 for non-overlapping pools)
#       * reduces over C (sum over channels)
#   - all fused ops share the same offs/mask (optimization-stage requirement).
# ---------------------------------------------


@triton.jit
def fused_maxpool_sum3d_kernel(
    x_ptr, y_ptr,
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    K: tl.constexpr,          # effective kernel size (cubic)
    STRIDE: tl.constexpr,     # effective stride (cubic)
    C_CONST: tl.constexpr,    # number of channels (compile-time for unrolling)
    BLOCK: tl.constexpr,      # block size (power of 2)
):
    # 1D grid over final output [N, 1, D_out, H_out, W_out]
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    P = N * D_out * H_out * W_out
    mask = offs < P

    # Decode offs -> (n, oz, oy, ox)  (no channel dim; it's reduced)
    WH_out = W_out
    HW_out = H_out * W_out
    DHW_out = D_out * HW_out

    tmp = offs
    ox = tmp % WH_out
    tmp = tmp // WH_out
    oy = tmp % H_out
    tmp = tmp // H_out
    oz = tmp % D_out
    n = tmp // D_out

    # Base output pointer (channel dimension is 1)
    y_ptrs = (
        y_ptr
        + n * y_stride_n
        + oz * y_stride_d
        + oy * y_stride_h
        + ox * y_stride_w
    )

    # Precompute starting indices in input for this output position
    iz0 = oz * STRIDE
    iy0 = oy * STRIDE
    ix0 = ox * STRIDE

    # Accumulator over channels
    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    # Reduction over channels, fused with spatial maxpool
    # All computations are indexed from the same (offs, mask) pair.
    for c in range(C_CONST):
        # Max over the K^3 window for this channel
        max_val = tl.full((BLOCK,), -1e30, dtype=tl.float32)

        # Base pointer for (n, c, :, :, :)
        base_nc = (
            x_ptr
            + n * x_stride_n
            + c * x_stride_c
        )

        # Unrolled 3 nested loops over K (compile-time known)
        for kz in range(K):
            iz = iz0 + kz
            base_z = base_nc + iz * x_stride_d
            for ky in range(K):
                iy = iy0 + ky
                base_y = base_z + iy * x_stride_h
                for kx in range(K):
                    ix = ix0 + kx
                    in_ptrs = base_y + ix * x_stride_w
                    vals = tl.load(in_ptrs, mask=mask, other=-1e30)
                    max_val = tl.maximum(max_val, vals)

        acc += max_val

    # Store final sum over channels
    tl.store(y_ptrs, acc, mask=mask)


def fused_two_maxpool_and_sum_triton(x, kernel1, stride1, kernel2, stride2):
    """
    Fuses:
      MaxPool3d(kernel1, stride1) -> MaxPool3d(kernel2, stride2) -> sum over channels
    into a single effective pool + channel-reduction.

    x: [N, C, D_in, H_in, W_in]
    Returns: [N, 1, D_out, H_out, W_out]
    """
    assert x.is_contiguous()
    device = x.device
    N, C, D_in, H_in, W_in = x.shape

    # Effective kernel and stride for two non-overlapping pools in sequence.
    # 1D derivation:
    #   D1 = (D_in - K1) / S1 + 1
    #   D2 = (D1  - K2) / S2 + 1
    #   => D2 = (D_in - (S1*(K2-1) + K1)) / (S1*S2) + 1
    K1, S1 = kernel1, stride1
    K2, S2 = kernel2, stride2
    K_eff = S1 * (K2 - 1) + K1
    S_eff = S1 * S2

    # Final output shape (matches two-step pooling)
    D_out = (D_in - K_eff) // S_eff + 1
    H_out = (H_in - K_eff) // S_eff + 1
    W_out = (W_in - K_eff) // S_eff + 1

    y = torch.empty((N, 1, D_out, H_out, W_out), device=device, dtype=x.dtype)

    x_strides = x.stride()
    y_strides = y.stride()

    P = N * D_out * H_out * W_out
    BLOCK = 128  # power-of-2, good occupancy on 4090
    grid = (triton.cdiv(P, BLOCK),)

    fused_maxpool_sum3d_kernel[grid](
        x, y,
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3], x_strides[4],
        y_strides[0], y_strides[1], y_strides[2], y_strides[3], y_strides[4],
        K=K_eff, STRIDE=S_eff, C_CONST=C, BLOCK=BLOCK,
    )
    return y


# ---------------------------------------------
# (Optional) Original kernels kept for completeness / generic usage
# ---------------------------------------------


@triton.jit
def maxpool3d_kernel(
    x_ptr, y_ptr,
    N, C,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    K: tl.constexpr,          # kernel size (cubic)
    STRIDE: tl.constexpr,     # stride (cubic)
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    Q = N * C * D_out * H_out * W_out
    mask = offs < Q

    # Decode offs -> (n, c, oz, oy, ox)
    WH_out = W_out
    HW_out = H_out * W_out
    DHW_out = D_out * HW_out

    tmp = offs
    ox = tmp % WH_out
    tmp = tmp // WH_out
    oy = tmp % H_out
    tmp = tmp // H_out
    oz = tmp % D_out
    tmp = tmp // D_out
    c = tmp % C
    n = tmp // C

    # Base pointer for output
    y_ptrs = (
        y_ptr
        + n * y_stride_n
        + c * y_stride_c
        + oz * y_stride_d
        + oy * y_stride_h
        + ox * y_stride_w
    )

    # Initialize max with large negative
    max_val = tl.full((BLOCK,), -1e30, dtype=tl.float32)

    # Starting indices in input
    iz0 = oz * STRIDE
    iy0 = oy * STRIDE
    ix0 = ox * STRIDE

    for kz in range(K):
        iz = iz0 + kz
        for ky in range(K):
            iy = iy0 + ky
            for kx in range(K):
                ix = ix0 + kx

                in_ptrs = (
                    x_ptr
                    + n * x_stride_n
                    + c * x_stride_c
                    + iz * x_stride_d
                    + iy * x_stride_h
                    + ix * x_stride_w
                )

                vals = tl.load(in_ptrs, mask=mask, other=-1e30)
                max_val = tl.maximum(max_val, vals)

    tl.store(y_ptrs, max_val, mask=mask)


def maxpool3d_triton(x, kernel_size, stride):
    """
    x: [N, C, D_in, H_in, W_in]
    """
    assert x.is_contiguous()
    device = x.device
    N, C, D_in, H_in, W_in = x.shape
    K = kernel_size
    S = stride

    # Standard PyTorch MaxPool3d output size (pad=0, dilation=1)
    D_out = (D_in - K) // S + 1
    H_out = (H_in - K) // S + 1
    W_out = (W_in - K) // S + 1

    y = torch.empty((N, C, D_out, H_out, W_out), device=device, dtype=x.dtype)

    x_strides = x.stride()
    y_strides = y.stride()

    Q = N * C * D_out * H_out * W_out
    BLOCK = 128
    grid = (triton.cdiv(Q, BLOCK),)

    maxpool3d_kernel[grid](
        x, y,
        N, C,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3], x_strides[4],
        y_strides[0], y_strides[1], y_strides[2], y_strides[3], y_strides[4],
        K, S,
        BLOCK=BLOCK,
    )
    return y


@triton.jit
def sum_channels_kernel(
    x_ptr, y_ptr,
    N, C,
    D, H, W,
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w,
    C_CONST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    P = N * D * H * W
    mask = offs < P

    # Decode offs -> (n, z, y, x)
    WH = W
    HW = H * W
    DHW = D * HW

    tmp = offs
    x_idx = tmp % W
    tmp = tmp // W
    y_idx = tmp % H
    tmp = tmp // H
    z_idx = tmp % D
    n_idx = tmp // D

    # Base output pointer (channel dimension is 1)
    y_ptrs = (
        y_ptr
        + n_idx * y_stride_n
        + z_idx * y_stride_d
        + y_idx * y_stride_h
        + x_idx * y_stride_w
    )

    # Reduction over channels
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in range(C_CONST):
        in_ptrs = (
            x_ptr
            + n_idx * x_stride_n
            + c * x_stride_c
            + z_idx * x_stride_d
            + y_idx * x_stride_h
            + x_idx * x_stride_w
        )
        vals = tl.load(in_ptrs, mask=mask, other=0.0)
        acc += vals

    tl.store(y_ptrs, acc, mask=mask)


def sum_channels_triton(x):
    """
    x: [N, C, D, H, W] -> sum over C, keepdim=True => [N, 1, D, H, W]
    """
    assert x.is_contiguous()
    device = x.device
    N, C, D, H, W = x.shape

    y = torch.empty((N, 1, D, H, W), device=device, dtype=x.dtype)

    x_strides = x.stride()
    y_strides = y.stride()

    P = N * D * H * W
    BLOCK = 128
    grid = (triton.cdiv(P, BLOCK),)

    sum_channels_kernel[grid](
        x, y,
        N, C,
        D, H, W,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3], x_strides[4],
        y_strides[0], y_strides[1], y_strides[2], y_strides[3], y_strides[4],
        C, BLOCK=BLOCK,
    )
    return y


# ---------------------------------------------
# High-level Model
# ---------------------------------------------


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for:
      ConvTranspose3d -> MaxPool3d(kernel=2) -> MaxPool3d(kernel=3) -> sum over channels

    The two MaxPool3d layers and the channel-sum are fused into a single
    high-throughput Triton kernel that:
      - uses a single 1D grid over the final output tensor
      - shares the same offsets + mask for all fused operations
      - dramatically reduces global memory traffic.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weight layout: [C_in, C_out, K, K, K]
        k = kernel_size
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, k, k, k)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # Ensure GPU and contiguous before Triton
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias

        # ConvTranspose3d
        x = conv_transpose3d_triton(
            x, weight, bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        # Fused: MaxPool3d(kernel=2, stride=2) -> MaxPool3d(kernel=3, stride=3) -> sum over channels
        x = fused_two_maxpool_and_sum_triton(
            x,
            kernel1=2, stride1=2,
            kernel2=3, stride2=3,
        )
        return x
