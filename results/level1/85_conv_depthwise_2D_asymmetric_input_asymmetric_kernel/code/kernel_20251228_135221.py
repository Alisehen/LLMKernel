import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_kernel(
    x_ptr,        # float32[N, C, H, W]
    w_ptr,        # float32[C, Kh, Kw]
    b_ptr,        # float32[C] (optional, controlled by HAS_BIAS)
    y_ptr,        # float32[N, C, H_out, W_out]
    N, C, H, W,
    H_out, W_out,
    Kh, Kw,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    BLOCK_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # program ids
    pid_row = tl.program_id(axis=0)  # over N*C*H_out
    pid_w   = tl.program_id(axis=1)  # over W_out tiles

    # decode (n, c, oh) from pid_row
    NC_Ho = C * H_out
    n = pid_row // NC_Ho
    rem = pid_row % NC_Ho
    c = rem // H_out
    oh = rem % H_out

    # if pid_row >= N*C*H_out, mask out all work
    full_rows = N * C * H_out
    row_mask = pid_row < full_rows

    # compute W offsets for this program
    w_block_start = pid_w * BLOCK_W
    offs_w = w_block_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_out
    mask_w = mask_w & row_mask

    # initialize accumulator
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # precompute some bases
    base_nc = (n * C + c) * H * W
    h_start = oh * stride_h - pad_h

    # iterate over kernel height
    for kh in range(0, Kh):
        ih = h_start + kh * dil_h
        h_in_bounds = (ih >= 0) & (ih < H)

        # base index in input for this (n, c, ih, 0)
        base_h = base_nc + ih * W

        # iterate over kernel width
        for kw in range(0, Kw):
            iw = offs_w * stride_w - pad_w + kw * dil_w
            w_in_bounds = (iw >= 0) & (iw < W)
            mask = mask_w & w_in_bounds & h_in_bounds

            # load input values
            x_idx = base_h + iw
            x_vals = tl.load(x_ptr + x_idx, mask=mask, other=0.0)

            # load weight scalar
            w_offset = (c * Kh + kh) * Kw + kw
            w_val = tl.load(w_ptr + w_offset)

            acc += x_vals * w_val

    if HAS_BIAS:
        bias_val = tl.load(b_ptr + c)
        acc = acc + bias_val

    # store result
    out_base = ((n * C + c) * H_out + oh) * W_out
    y_idx = out_base + offs_w
    tl.store(y_ptr + y_idx, acc, mask=mask_w)


def depthwise_conv2d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,  # [C, Kh, Kw]
    bias: torch.Tensor | None,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dil_h: int,
    dil_w: int,
) -> torch.Tensor:
    """
    Depthwise 2D convolution (groups = in_channels) using Triton.

    x:      [N, C, H, W], contiguous, CUDA
    weight: [C, Kh, Kw], contiguous, same device/dtype as x
    bias:   [C] or None
    """
    assert x.is_cuda, "Triton kernels require CUDA tensors"
    assert x.dtype == torch.float32, "Kernel currently supports float32"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C, H, W = x.shape
    Cw, Kh, Kw = weight.shape
    assert Cw == C, "Weight channels must match input channels for depthwise conv"

    # compute output dimensions (same as PyTorch Conv2d)
    H_out = (H + 2 * pad_h - dil_h * (Kh - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (Kw - 1) - 1) // stride_w + 1

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_W = 128  # power-of-2 block size

    total_rows = N * C * H_out
    grid = lambda meta: (
        total_rows,
        triton.cdiv(W_out, meta["BLOCK_W"]),
    )

    HAS_BIAS = bias is not None
    b_ptr = bias if bias is not None else y  # dummy pointer if bias is absent

    depthwise_conv2d_kernel[grid](
        x,
        weight,
        b_ptr,
        y,
        N,
        C,
        H,
        W,
        H_out,
        W_out,
        Kh,
        Kw,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        BLOCK_W=BLOCK_W,
        HAS_BIAS=HAS_BIAS,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Depthwise 2D convolution implemented with a high-performance Triton kernel.
    Mirrors the behavior of:
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                  dilation, groups=in_channels, bias=bias)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_h: int,
        kernel_size_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        padding_h: int = 0,
        padding_w: int = 0,
        dilation_h: int = 1,
        dilation_w: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # Depthwise: one filter per input channel
        assert (
            in_channels == out_channels
        ), "Depthwise conv expects in_channels == out_channels"
        assert (
            groups == in_channels
        ), "Depthwise conv expects groups == in_channels"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size_h = kernel_size_h
        self.kernel_size_w = kernel_size_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.padding_h = padding_h
        self.padding_w = padding_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups

        # Match Conv2d(depthwise) parameter shapes: [C, 1, Kh, Kw]
        weight = torch.empty(
            out_channels, 1, kernel_size_h, kernel_size_w
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5)) if hasattr(nn.init, "kaiming_uniform_") else None
        self.weight = nn.Parameter(weight)

        if bias:
            b = torch.empty(out_channels)
            fan_in = in_channels * kernel_size_h * kernel_size_w
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(b, -bound, bound)
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert weight to [C, Kh, Kw] for Triton kernel
        w = self.weight.view(self.out_channels, self.kernel_size_h, self.kernel_size_w)
        return depthwise_conv2d_triton(
            x,
            w,
            self.bias,
            self.stride_h,
            self.stride_w,
            self.padding_h,
            self.padding_w,
            self.dilation_h,
            self.dilation_w,
        )
