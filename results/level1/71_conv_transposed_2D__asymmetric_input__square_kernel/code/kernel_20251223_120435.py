# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Optimized implicit-GEMM kernel for ConvTranspose2d (stride=1, groups=1, output_padding=0)
# 3D grid = (H_out*W_out tiles, C_out tiles, batch)
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Baseline / conservative tile (original baseline)
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Square tile: better balance between HW and C_out
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Larger tile, reduced warps to improve occupancy on some shapes
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Larger tile with more warps for very compute-heavy regimes
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["C_IN", "C_OUT", "H_out", "W_out"],
)
@triton.jit
def conv_transpose2d_implicit_gemm_kernel(
    x_ptr,  # [N, C_in, H_in, W_in]
    w_ptr,  # [C_in, C_out, K, K]  (groups=1)
    b_ptr,  # [C_out] or dummy
    y_ptr,  # [N, C_out, H_out, W_out]
    M,      # = H_out * W_out  (flattened output spatial dim per batch)
    C_IN,
    C_OUT,
    H_in,
    W_in,
    H_out,
    W_out,
    PADDING: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid:
    #  - axis 0: tiles over output spatial HW per batch
    #  - axis 1: tiles over output channels
    #  - axis 2: batches
    pid_hw = tl.program_id(axis=0)
    pid_co = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    # Offsets in spatial (flattened HW) and channels
    offs_m = pid_hw * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_co * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M          # M == H_out * W_out
    mask_n = offs_n < C_OUT

    # Decode flattened HW index -> (ho, wo)
    ho = offs_m // W_out
    wo = offs_m % W_out

    # Batch index (scalar)
    n_idx = pid_n

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_total = C_IN * K * K

    # Reduction over K_total = C_IN * K * K in blocks of BLOCK_K
    for k_start in range(0, K_total, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_idx < K_total

        # Decode k_idx -> (ci, kh, kw)
        ci = k_idx // (K * K)
        kk = k_idx % (K * K)
        kh = kk // K
        kw = kk % K

        # Compute input spatial coordinates for stride=1 case:
        # h_in = ho + PADDING - kh
        # w_in = wo + PADDING - kw
        ho_b = ho[:, None]        # [BLOCK_M, 1]
        wo_b = wo[:, None]
        kh_b = kh[None, :]        # [1, BLOCK_K]
        kw_b = kw[None, :]

        h_in = ho_b + PADDING - kh_b  # [BLOCK_M, BLOCK_K]
        w_in = wo_b + PADDING - kw_b  # [BLOCK_M, BLOCK_K]

        # Bounds mask for input
        in_bounds = (
            (h_in >= 0) & (h_in < H_in) &
            (w_in >= 0) & (w_in < W_in)
        )

        # Full mask for loading X
        mask_x = (
            mask_m[:, None] &
            mask_k[None, :] &
            in_bounds
        )

        # Compute X offsets: ((n * C_IN + ci) * H_in + h_in) * W_in + w_in
        n_factor = n_idx * (C_IN * H_in * W_in)   # scalar
        ci_factor = ci * (H_in * W_in)            # [BLOCK_K]
        hw_factor = h_in * W_in + w_in            # [BLOCK_M, BLOCK_K]

        x_offsets = (
            n_factor +
            ci_factor[None, :] +
            hw_factor
        )

        x_vals = tl.load(x_ptr + x_offsets, mask=mask_x, other=0.0)
        x_vals = x_vals.to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Compute W offsets:
        # layout: w[ci, co, kh, kw] contiguous
        # offset = ((ci * C_OUT + co) * K + kh) * K + kw
        co = offs_n                            # [BLOCK_N]
        base_ci = ci * (C_OUT * K * K)         # [BLOCK_K]
        base_k = kh * K + kw                   # [BLOCK_K]
        co_contrib = co * (K * K)              # [BLOCK_N]

        w_offsets = (
            base_ci[:, None] +
            co_contrib[None, :] +
            base_k[:, None]
        )

        mask_w = mask_k[:, None] & mask_n[None, :]

        w_vals = tl.load(w_ptr + w_offsets, mask=mask_w, other=0.0)
        w_vals = w_vals.to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # GEMM-style accumulation: [M, K] x [K, N] -> [M, N]
        acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        bias = bias.to(tl.float32)
        acc += bias[None, :]

    # Store results back to y[n, co, ho, wo]
    C_out_HW = C_OUT * H_out * W_out

    batch_factor = n_idx * C_out_HW               # scalar
    co_factor = offs_n * (H_out * W_out)          # [BLOCK_N]
    hw_factor = ho * W_out + wo                   # [BLOCK_M]

    y_offsets = (
        batch_factor +
        co_factor[None, :] +
        hw_factor[:, None]
    )

    mask_y = mask_m[:, None] & mask_n[None, :]

    tl.store(y_ptr + y_offsets, acc.to(tl.float32), mask=mask_y)


# -----------------------------------------------------------------------------
# Fallback kernel: generic but less optimized (handles arbitrary stride, padding, groups)
# Grid is 3D: (batch, C_out tiles, HW tiles)
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Baseline behavior (matches prior implicit defaults)
        triton.Config({}, num_warps=4, num_stages=2),
        # Alternative with more warps for some shapes
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["C_IN", "C_OUT", "H_out", "W_out"],
)
@triton.jit
def conv_transpose2d_naive_kernel(
    x_ptr,  # [N, C_IN, H_in, W_in]
    w_ptr,  # [C_IN, C_OUT/GROUPS, K, K]
    b_ptr,  # [C_OUT] or dummy
    y_ptr,  # [N, C_OUT, H_out, W_out]
    N,
    H_in,
    W_in,
    H_out,
    W_out,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    OUTPUT_PADDING: tl.constexpr,  # kept for interface parity; not used explicitly
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    GROUPS: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_co = tl.program_id(axis=1)
    pid_hw = tl.program_id(axis=2)

    # Batch index
    n = pid_n

    # Tile of output channels and spatial positions
    co_block_start = pid_co * BLOCK_CO
    hw_block_start = pid_hw * BLOCK_HW

    co_idxs = co_block_start + tl.arange(0, BLOCK_CO)
    hw_idxs = hw_block_start + tl.arange(0, BLOCK_HW)

    co_mask = co_idxs < C_OUT
    hw_mask = hw_idxs < (H_out * W_out)

    # Map flattened spatial indices -> (ho, wo)
    ho = hw_idxs // W_out
    wo = hw_idxs % W_out

    # Initialize accumulator in FP32 for numerical stability
    out = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32)

    # Precompute group-related constants
    Gin = C_IN // GROUPS
    Gout = C_OUT // GROUPS

    # Channel group id for each output channel in this tile
    group_co = co_idxs // Gout
    co_within_group = co_idxs % Gout

    # Batch strides
    x_batch_stride = C_IN * H_in * W_in
    y_batch_stride = C_OUT * H_out * W_out
    x_hw_stride = W_in
    y_hw_stride = W_out
    x_c_stride = H_in * W_in
    y_c_stride = H_out * W_out

    # Base offsets for this batch
    x_batch_offset = n * x_batch_stride
    y_batch_offset = n * y_batch_stride

    # Reduction over input channels and kernel elements
    for ci in range(C_IN):
        group_ci = ci // Gin

        # Mask of output channels in this tile that connect to this input channel
        w_mask_co = co_mask & (group_co == group_ci)

        for kh in range(K):
            for kw in range(K):
                # Compute candidate input coordinates for each output position
                num_h = ho + PADDING - kh
                num_w = wo + PADDING - kw

                mask = hw_mask & (num_h >= 0) & (num_w >= 0)
                # Stride alignment
                if STRIDE != 1:
                    mask &= (num_h % STRIDE == 0) & (num_w % STRIDE == 0)

                h_in = num_h // STRIDE
                w_in = num_w // STRIDE

                # Bounds check on input spatial indices
                mask &= (h_in < H_in) & (w_in < W_in)

                # Load input values for this (ci, kh, kw) across spatial tile
                x_offsets = (
                    x_batch_offset
                    + ci * x_c_stride
                    + h_in * x_hw_stride
                    + w_in
                )
                x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
                x_vals = x_vals.to(tl.float32)

                # Load weights for this input channel and kernel position
                w_offsets = (
                    ci * (Gout * K * K)
                    + co_within_group * (K * K)
                    + kh * K
                    + kw
                )
                w_vals = tl.load(w_ptr + w_offsets, mask=w_mask_co, other=0.0)
                w_vals = w_vals.to(tl.float32)

                # Outer product: [BLOCK_CO, 1] * [1, BLOCK_HW] -> [BLOCK_CO, BLOCK_HW]
                contrib = w_vals[:, None] * x_vals[None, :]
                out += contrib

    # Add bias if present
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + co_idxs, mask=co_mask, other=0.0)
        bias_vals = bias_vals.to(tl.float32)
        out += bias_vals[:, None]

    # Store results
    store_mask = co_mask[:, None] & hw_mask[None, :]

    y_offsets = (
        y_batch_offset
        + co_idxs[:, None] * y_c_stride
        + ho[None, :] * y_hw_stride
        + wo[None, :]
    )

    tl.store(y_ptr + y_offsets, out.to(tl.float32), mask=store_mask)


# -----------------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------------

def triton_conv_transpose2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    output_padding: int,
    groups: int,
) -> torch.Tensor:
    """
    High-performance ConvTranspose2d (2D transposed convolution) using Triton.

    Args:
        x: Input tensor of shape (N, C_in, H_in, W_in), float32, CUDA.
        weight: Weight tensor of shape (C_in, C_out // groups, K, K), float32, CUDA.
        bias: Optional bias tensor of shape (C_out,), float32, CUDA.
        stride, padding, output_padding, groups: Same semantics as nn.ConvTranspose2d,
            with single integer for both spatial dimensions.

    Returns:
        Output tensor of shape (N, C_out, H_out, W_out), float32, CUDA.
    """
    assert x.is_cuda and weight.is_cuda, "Input and weight must be CUDA tensors"
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, "Only float32 supported for now"
    if bias is not None:
        assert bias.is_cuda and bias.dtype == torch.float32

    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out_per_group, K, K2 = weight.shape
    assert C_in == C_in_w, "Weight in_channels must match input channels"
    assert K == K2, "Only square kernels are supported"
    assert isinstance(stride, int) and isinstance(padding, int) and isinstance(output_padding, int)
    assert groups > 0
    C_out = C_out_per_group * groups

    # Output spatial size following PyTorch's ConvTranspose2d formula
    H_out = (H_in - 1) * stride - 2 * padding + K + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K + output_padding

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.float32)

    # Decide whether to use the optimized implicit-GEMM kernel
    use_implicit_gemm = (stride == 1) and (groups == 1) and (output_padding == 0)

    if use_implicit_gemm:
        # Implicit-GEMM tiling configuration via autotune
        M = H_out * W_out  # per-batch spatial size

        def grid(meta):
            # 3D grid: (HW tiles, C_out tiles, batch)
            return (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(C_out, meta["BLOCK_N"]),
                N,
            )

        conv_transpose2d_implicit_gemm_kernel[grid](
            x,
            # For groups=1, weight layout is [C_in, C_out, K, K]
            weight,
            bias if bias is not None else x,  # dummy pointer if no bias
            y,
            M,
            C_in,
            C_out,
            H_in,
            W_in,
            H_out,
            W_out,
            PADDING=padding,
            K=K,
            HAS_BIAS=bias is not None,
        )
    else:
        # Fallback to the generic kernel that supports arbitrary stride / groups
        BLOCK_CO = 32  # tile over output channels (power of 2)
        BLOCK_HW = 64  # tile over spatial positions (power of 2)

        def grid(meta):
            # 3D grid: (batch, C_out tiles, HW tiles)
            return (
                N,
                triton.cdiv(C_out, BLOCK_CO),
                triton.cdiv(H_out * W_out, BLOCK_HW),
            )

        conv_transpose2d_naive_kernel[grid](
            x,
            weight,
            bias if bias is not None else x,  # dummy pointer if no bias
            y,
            N,
            H_in,
            W_in,
            H_out,
            W_out,
            STRIDE=stride,
            PADDING=padding,
            OUTPUT_PADDING=output_padding,
            C_IN=C_in,
            C_OUT=C_out,
            GROUPS=groups,
            K=K,
            HAS_BIAS=bias is not None,
            BLOCK_CO=BLOCK_CO,
            BLOCK_HW=BLOCK_HW,
        )

    return y


# -----------------------------------------------------------------------------
# nn.Module wrapper
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Triton-optimized replacement for ConvTranspose2d-based models.

    Performs a transposed 2D convolution with a square kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # Use PyTorch module only as parameter container
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias
        return triton_conv_transpose2d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
        )
