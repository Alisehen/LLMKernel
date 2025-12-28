# <complete ModelNew code with optimized Triton kernels>
import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv_transpose2d_fwd_kernel(
    x_ptr,  # [N, C_in, H_in, W_in]
    w_ptr,  # [C_in, C_out, K, K]
    b_ptr,  # [C_out] (ignored if has_bias=False)
    y_ptr,  # [N, C_out, H_out, W_out]
    N,      # batch size
    H_in,
    W_in,
    C_out,
    H_out,
    W_out,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    has_bias: tl.constexpr,
    C_IN: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_co = tl.program_id(axis=1)
    pid_hw = tl.program_id(axis=2)

    # Tile indices along channels-out and flattened spatial dimension
    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    HW_out_total = H_out * W_out

    mask_co = co_offsets < C_out
    mask_hw = hw_offsets < HW_out_total

    # Map flattened hw_offsets -> (ho, wo)
    ho = hw_offsets // W_out
    wo = hw_offsets - ho * W_out

    # Accumulator for output tile [BLOCK_HW, BLOCK_CO]
    acc = tl.zeros((BLOCK_HW, BLOCK_CO), dtype=tl.float32)

    # Loop over C_in and kernel spatial dimensions (all compile-time for efficiency)
    for cin in range(0, C_IN):
        x_base = ((pid_n * C_IN + cin) * H_in) * W_in

        for ky in range(0, KERNEL_SIZE):
            # Compute contributing input row index for each output row
            tmp_h = ho + padding - ky * dilation
            h_in = tmp_h // stride
            valid_h = (tmp_h >= 0) & (h_in < H_in) & (tmp_h == h_in * stride)

            for kx in range(0, KERNEL_SIZE):
                # Compute contributing input col index for each output col
                tmp_w = wo + padding - kx * dilation
                w_in = tmp_w // stride
                valid_w = (tmp_w >= 0) & (w_in < W_in) & (tmp_w == w_in * stride)

                mask_x = mask_hw & valid_h & valid_w

                x_offsets = x_base + h_in * W_in + w_in
                x_vals = tl.load(x_ptr + x_offsets, mask=mask_x, other=0.0)  # [BLOCK_HW]

                # Load weights w[cin, co, ky, kx] for this cin, ky, kx and co tile
                w_offsets = (((cin * C_out + co_offsets) * KERNEL_SIZE + ky) * KERNEL_SIZE) + kx
                w_vals = tl.load(w_ptr + w_offsets, mask=mask_co, other=0.0)  # [BLOCK_CO]

                # Outer product: [BLOCK_HW, 1] * [1, BLOCK_CO] -> [BLOCK_HW, BLOCK_CO]
                x_vals_2d = x_vals[:, None]
                w_vals_2d = w_vals[None, :]
                acc += x_vals_2d * w_vals_2d

    # Add bias if present
    if has_bias:
        b_vals = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0)  # [BLOCK_CO]
        acc += b_vals[None, :]

    # Store results
    n_offset = pid_n * C_out * H_out * W_out
    ho_2d = ho[:, None]
    wo_2d = wo[:, None]
    co_2d = co_offsets[None, :]

    y_offsets = n_offset + ((co_2d * H_out + ho_2d) * W_out + wo_2d)
    store_mask = mask_hw[:, None] & mask_co[None, :]

    tl.store(y_ptr + y_offsets, acc, mask=store_mask)


def triton_conv_transpose2d(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            stride: int,
                            padding: int,
                            dilation: int) -> torch.Tensor:
    """
    x:       [N, C_in, H_in, W_in]
    weight:  [C_in, C_out, K, K]  (ConvTranspose2d layout)
    bias:    [C_out] or None
    """
    assert x.is_cuda and weight.is_cuda, "Triton conv_transpose2d only supports CUDA tensors"
    x = x.contiguous()
    weight = weight.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out, K, K2 = weight.shape
    assert C_in_w == C_in and K == K2, "Weight shape must be [C_in, C_out, K, K]"

    # Output size formula for ConvTranspose2d (output_padding=0)
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    if bias is not None:
        bias = bias.contiguous()
        has_bias = True
        b_ptr = bias
    else:
        has_bias = False
        # Dummy pointer; never accessed when has_bias=False
        b_ptr = weight

    BLOCK_CO = 32
    BLOCK_HW = 64

    grid = lambda meta: (
        N,
        triton.cdiv(C_out, meta["BLOCK_CO"]),
        triton.cdiv(H_out * W_out, meta["BLOCK_HW"]),
    )

    conv_transpose2d_fwd_kernel[grid](
        x, weight, b_ptr, y,
        N, H_in, W_in,
        C_out, H_out, W_out,
        stride=stride,
        padding=padding,
        dilation=dilation,
        has_bias=has_bias,
        C_IN=C_in,
        KERNEL_SIZE=K,
        BLOCK_CO=BLOCK_CO,
        BLOCK_HW=BLOCK_HW,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for nn.ConvTranspose2d with square kernel,
    supporting stride, padding, dilation, and optional bias.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False):
        super().__init__()
        # Keep a real ConvTranspose2d module so external code can copy weights/bias into it.
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias
        return triton_conv_transpose2d(
            x, w, b, self.stride, self.padding, self.dilation
        )
