import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr,  # [N, C_IN, H_in, W_in]
    w_ptr,  # [C_IN, C_OUT/GROUPS, K, K]
    b_ptr,  # [C_OUT] or dummy
    y_ptr,  # [N, C_OUT, H_out, W_out]
    N, H_in, W_in,
    H_out, W_out,
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

        # Skip work if no channel in this tile belongs to this group
        # (masking-based: if all False, loads/stores are effectively no-op)
        for kh in range(K):
            for kw in range(K):
                # Compute candidate input coordinates for each output position
                num_h = ho + PADDING - kh
                num_w = wo + PADDING - kw

                mask = hw_mask & (num_h >= 0) & (num_w >= 0)
                # Stride alignment
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

    # Store results (cast back to original dtype = fp32 here)
    store_mask = co_mask[:, None] & hw_mask[None, :]

    y_offsets = (
        y_batch_offset
        + co_idxs[:, None] * y_c_stride
        + ho[None, :] * y_hw_stride
        + wo[None, :]
    )

    tl.store(y_ptr + y_offsets, out.to(tl.float32), mask=store_mask)


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

    # Tiling configuration
    BLOCK_CO = 32  # tile over output channels
    BLOCK_HW = 64  # tile over spatial positions

    def grid(meta):
        return (
            N,
            triton.cdiv(C_out, meta["BLOCK_CO"]),
            triton.cdiv(H_out * W_out, meta["BLOCK_HW"]),
        )

    conv_transpose2d_kernel[grid](
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


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for the given ConvTranspose2d-based Model.

    Performs a transposed 2D convolution with asymmetric input and a square kernel.
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
