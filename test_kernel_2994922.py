import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_fwd_kernel(
    x_ptr,          # *f32, [N, C_IN, L_IN]
    w_ptr,          # *f32, [C_IN, C_OUT, K]
    b_ptr,          # *f32, [C_OUT] or dummy
    y_ptr,          # *f32, [N, C_OUT, L_OUT]
    N,              # int32
    L_IN,           # int32
    L_OUT,          # int32
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    DILATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # program ids
    pid_l = tl.program_id(axis=0)   # along output length
    pid_nc = tl.program_id(axis=1)  # over batch * out_channels

    # decode (n, co) from pid_nc
    n = pid_nc // C_OUT
    co = pid_nc % C_OUT

    # offsets along output length
    l_out_start = pid_l * BLOCK_L
    offs_l = l_out_start + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L_OUT

    # accumulator in fp32 for better numeric stability
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # precompute constants for pointer arithmetic
    cin_stride = L_IN
    n_stride = C_IN * L_IN
    cout_stride_w = K
    cin_stride_w = C_OUT * K
    cout_stride_y = L_OUT
    n_stride_y = C_OUT * L_OUT

    # loop over input channels and kernel positions
    for ci in range(0, C_IN):
        for k in range(0, K):
            # For transposed conv, relation between l_out and l_in:
            # l_out = l_in * STRIDE - PADDING + k * DILATION
            # => l_in = (l_out + PADDING - k * DILATION) / STRIDE (integer, in-range)
            v = offs_l + PADDING - k * DILATION

            # check divisibility by STRIDE
            rem = v % STRIDE
            mask_div = rem == 0

            l_in = v // STRIDE
            mask_in_range = (l_in >= 0) & (l_in < L_IN)

            mask = mask_l & mask_div & mask_in_range

            # avoid out-of-bounds pointer arithmetic for masked elements
            safe_l_in = tl.where(mask, l_in, 0)

            x_offsets = n * n_stride + ci * cin_stride + safe_l_in
            x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

            w_offset = ci * cin_stride_w + co * cout_stride_w + k
            w_val = tl.load(w_ptr + w_offset)

            acc += x_vals * w_val

    if HAS_BIAS:
        b_val = tl.load(b_ptr + co)
        acc += b_val

    y_offsets = n * n_stride_y + co * cout_stride_y + offs_l
    tl.store(y_ptr + y_offsets, acc, mask=mask_l)


def triton_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
) -> torch.Tensor:
    """
    x:       [N, C_in, L_in]
    weight:  [C_in, C_out, K]
    bias:    [C_out] or None
    """
    # Fallback to PyTorch if not CUDA
    if not x.is_cuda:
        return torch.nn.functional.conv_transpose1d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation
        )

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, L_in = x.shape
    C_in_w, C_out, K = weight.shape
    assert C_in_w == C_in, "Weight C_in must match input C_in"

    # Output length for ConvTranspose1d (output_padding = 0)
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    y = torch.empty((N, C_out, L_out), device=x.device, dtype=x.dtype)

    # Decide kernel launch config
    BLOCK_L = 256  # power-of-2 as required
    grid = lambda meta: (
        triton.cdiv(L_out, meta["BLOCK_L"]),
        max(1, N * C_out),
    )

    HAS_BIAS = bias is not None
    b_ptr = bias if HAS_BIAS else y  # dummy ptr if no bias

    conv_transpose1d_fwd_kernel[grid](
        x,
        weight,
        b_ptr,
        y,
        N,
        L_in,
        L_out,
        C_IN=C_in,
        C_OUT=C_out,
        K=K,
        STRIDE=stride,
        PADDING=padding,
        DILATION=dilation,
        HAS_BIAS=HAS_BIAS,
        BLOCK_L=BLOCK_L,
        num_warps=4,
    )

    return y


class ModelNew(nn.Module):
    """
    ConvTranspose1d implemented with a high-performance Triton kernel.
    API matches the given PyTorch Model.
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
        # Use PyTorch ConvTranspose1d only for parameter initialization
        ref = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(ref.weight.detach().clone())
        if bias:
            self.bias = nn.Parameter(ref.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
