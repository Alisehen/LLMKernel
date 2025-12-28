import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_fwd_kernel(
    x_ptr,            # *f32  [N, Cin, Lin]
    w_ptr,            # *f32  [Cin, Cout, K]
    b_ptr,            # *f32  [Cout] (ignored if HAS_BIAS=False)
    y_ptr,            # *f32  [N, Cout, Lout]
    N,                # int32
    L_in,             # int32
    L_out,            # int32
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    # 2D grid:
    #  pid_nl: over N * L_out (each program -> one (n, l_out))
    #  pid_cout: over tiles of C_out of size BLOCK_COUT
    pid_nl = tl.program_id(axis=0)
    pid_cout = tl.program_id(axis=1)

    total_nl = N * L_out
    mask_nl = pid_nl < total_nl

    # Decode (n, l_out) from linear id
    n = pid_nl // L_out
    l_out = pid_nl % L_out

    co_offsets = pid_cout * BLOCK_COUT + tl.arange(0, BLOCK_COUT)
    mask_co = co_offsets < COUT
    mask_store = mask_co & mask_nl

    # Initialize accumulator with bias (if any)
    acc = tl.zeros([BLOCK_COUT], dtype=tl.float32)
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0)
        acc += bias_vals

    # For each input channel and kernel position (compile-time loops)
    for ci in range(CIN):
        # Base factor for x indexing: (n * CIN + ci) * L_in
        x_channel_base = (n * CIN + ci) * L_in

        for k in range(K):
            # Compute corresponding input position l_in that contributes to l_out
            # l_out = l_in * stride - padding + k * dilation
            # => l_in = (l_out + padding - k * dilation) / stride
            raw = l_out + padding - k * dilation

            # Check raw is within valid range and divisible by stride
            is_nonneg = raw >= 0
            is_lt_max = raw < stride * L_in
            l_in = raw // stride
            is_div = raw == l_in * stride
            valid_lin = is_nonneg & is_lt_max & is_div & mask_nl

            # Load x[n, ci, l_in] if valid, else 0
            x_idx = x_channel_base + l_in
            x_val = tl.load(x_ptr + x_idx, mask=valid_lin, other=0.0)

            # Load weight vector w[ci, co_offsets, k]
            w_idx = (ci * COUT + co_offsets) * K + k
            w_vals = tl.load(w_ptr + w_idx, mask=mask_co, other=0.0)

            # FMA accumulate: acc += x_val * w_vals
            acc += x_val * w_vals

    # Store result y[n, co_offsets, l_out]
    y_idx = (n * COUT + co_offsets) * L_out + l_out
    tl.store(y_ptr + y_idx, acc, mask=mask_store)


def triton_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """
    x:       (N, Cin, L_in)
    weight:  (Cin, Cout, K)
    bias:    (Cout,) or None
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == weight.dtype == torch.float32, "Only float32 is supported"

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, Cin, L_in = x.shape
    w_Cin, Cout, K = weight.shape
    assert w_Cin == Cin, "Weight Cin must match input Cin"

    # PyTorch ConvTranspose1d output length formula (no output_padding)
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    y = torch.empty((N, Cout, L_out), device=x.device, dtype=x.dtype)

    BLOCK_COUT = 64  # power-of-2, good for vectorized channels

    grid = lambda meta: (
        N * L_out,  # pid_nl
        triton.cdiv(Cout, meta["BLOCK_COUT"]),  # pid_cout
    )

    conv_transpose1d_fwd_kernel[grid](
        x,
        weight,
        bias if bias is not None else y,  # dummy if no bias (unused when HAS_BIAS=False)
        y,
        N,
        L_in,
        L_out,
        stride=stride,
        padding=padding,
        dilation=dilation,
        CIN=Cin,
        COUT=Cout,
        K=K,
        HAS_BIAS=bias is not None,
        BLOCK_COUT=BLOCK_COUT,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Transposed 1D convolution implemented with a high-performance Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize parameters using PyTorch's ConvTranspose1d initialization
        ref = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.weight = nn.Parameter(ref.weight.detach())
        if bias:
            self.bias = nn.Parameter(ref.bias.detach())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
