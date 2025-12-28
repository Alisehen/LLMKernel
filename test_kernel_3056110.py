import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose3d_fwd_kernel(
    x_ptr,          # (N, C_IN, D_IN, H_IN, W_IN)
    w_ptr,          # (C_IN, C_OUT_PER_G, K_D, K_H, K_W)
    bias_ptr,       # (C_OUT,) or dummy when HAS_BIAS = False
    y_ptr,          # (N, C_OUT, D_OUT, H_OUT, W_OUT)
    N,
    D_IN, H_IN, W_IN,
    D_OUT, H_OUT, W_OUT,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(axis=0)  # over N*C_OUT
    pid_s = tl.program_id(axis=1)  # over D_OUT*H_OUT*W_OUT

    # total elements along each tiling dimension
    n_m = N * C_OUT
    n_s = D_OUT * H_OUT * W_OUT

    # offsets in flattened (N*C_OUT) and (D*H*W) spaces
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    mask_m = offs_m < n_m
    mask_s = offs_s < n_s

    # Decode m dimension -> (n, co)
    co = offs_m % C_OUT
    n = offs_m // C_OUT

    # Decode s dimension -> (d, h, w)
    DHW_OUT = H_OUT * W_OUT
    d = offs_s // DHW_OUT
    tmp = offs_s % DHW_OUT
    h = tmp // W_OUT
    w = tmp % W_OUT

    # 2D broadcasted indices
    n_2d = n[:, None]         # [BLOCK_M, 1]
    co_2d = co[:, None]       # [BLOCK_M, 1]
    d_2d = d[None, :]         # [1, BLOCK_S]
    h_2d = h[None, :]         # [1, BLOCK_S]
    w_2d_idx = w[None, :]     # [1, BLOCK_S]

    # group info per output channel
    C_OUT_PER_G = C_OUT // GROUPS
    C_IN_PER_G = C_IN // GROUPS
    g_co = co // C_OUT_PER_G  # [BLOCK_M]

    # initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_S), dtype=tl.float32)

    # optional bias
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + co, mask=mask_m, other=0.0)
        acc = acc + bias_vals[:, None]

    # base masks combining valid m and s
    base_mask = mask_m[:, None] & mask_s[None, :]

    K_SIZE = K_D * K_H * K_W

    # reduction over input channels and kernel volume
    for ci in range(C_IN):
        g_ci = ci // C_IN_PER_G

        # group connectivity mask per output channel
        g_match = g_co == g_ci  # [BLOCK_M]
        # co index within its group
        co_in_group = co - g_co * C_OUT_PER_G  # [BLOCK_M], meaningful where g_match

        # 2D mask combining m validity and group connectivity (no spatial yet)
        group_mask_2d = base_mask & g_match[:, None]

        for kz in range(K_D):
            # spatial relation in depth
            iz_nom = d + PAD_D - kz  # [BLOCK_S]
            mod_z = iz_nom % STRIDE_D
            is_int_z = mod_z == 0
            iz = iz_nom // STRIDE_D
            valid_iz = (iz >= 0) & (iz < D_IN)

            for ky in range(K_H):
                iy_nom = h + PAD_H - ky
                mod_y = iy_nom % STRIDE_H
                is_int_y = mod_y == 0
                iy = iy_nom // STRIDE_H
                valid_iy = (iy >= 0) & (iy < H_IN)

                for kx in range(K_W):
                    ix_nom = w + PAD_W - kx
                    mod_x = ix_nom % STRIDE_W
                    is_int_x = mod_x == 0
                    ix = ix_nom // STRIDE_W
                    valid_ix = (ix >= 0) & (ix < W_IN)

                    # spatial validity mask per column
                    mask_spatial_1d = (
                        is_int_z & valid_iz &
                        is_int_y & valid_iy &
                        is_int_x & valid_ix
                    )  # [BLOCK_S]
                    spatial_mask_2d = mask_spatial_1d[None, :]  # [1, BLOCK_S]

                    full_mask = group_mask_2d & spatial_mask_2d  # [BLOCK_M, BLOCK_S]

                    # Compute 2D input indices
                    iz_2d = iz[None, :]
                    iy_2d = iy[None, :]
                    ix_2d = ix[None, :]

                    in_offsets = (
                        (((n_2d * C_IN + ci) * D_IN + iz_2d) * H_IN + iy_2d) * W_IN
                        + ix_2d
                    )  # [BLOCK_M, BLOCK_S]

                    x_vals = tl.load(x_ptr + in_offsets, mask=full_mask, other=0.0)

                    # Load weights for current (ci, all co in tile, kz, ky, kx)
                    k_offset = (kz * K_H + ky) * K_W + kx
                    w_row_offsets = (ci * C_OUT_PER_G + co_in_group) * K_SIZE + k_offset
                    w_row = tl.load(
                        w_ptr + w_row_offsets,
                        mask=g_match & mask_m,
                        other=0.0,
                    )  # [BLOCK_M]

                    w_mat = w_row[:, None]  # [BLOCK_M, 1]

                    acc = acc + x_vals * w_mat

    # compute 2D output offsets (shape [BLOCK_M, BLOCK_S])
    out_offsets = (
        ((((n_2d * C_OUT) + co_2d) * D_OUT + d_2d) * H_OUT + h_2d) * W_OUT
        + w_2d_idx
    )

    tl.store(y_ptr + out_offsets, acc, mask=base_mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride=1,
    padding=0,
    output_padding=0,
    groups: int = 1,
) -> torch.Tensor:
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dtype == torch.float32, "Only float32 is supported"
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    # Normalize stride/padding/output_padding to 3-tuples
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding, output_padding)

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    out_pad_d, out_pad_h, out_pad_w = output_padding

    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_per_g, K_d, K_h, K_w = weight.shape
    assert C_in_w == C_in, "Weight in_channels must match input channels"
    C_out = C_out_per_g * groups

    # Output size (PyTorch ConvTranspose3d formula)
    D_out = (D_in - 1) * stride_d - 2 * pad_d + K_d + out_pad_d
    H_out = (H_in - 1) * stride_h - 2 * pad_h + K_h + out_pad_h
    W_out = (W_in - 1) * stride_w - 2 * pad_w + K_w + out_pad_w

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # BLOCK sizes must be power-of-2
    BLOCK_M = 32  # tile over N*C_out
    BLOCK_S = 64  # tile over D_out*H_out*W_out

    has_bias = bias is not None
    if bias is None:
        # dummy tensor to satisfy pointer argument
        bias = torch.empty(1, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(N * C_out, meta["BLOCK_M"]),
        triton.cdiv(D_out * H_out * W_out, meta["BLOCK_S"]),
    )

    conv_transpose3d_fwd_kernel[grid](
        x, weight, bias, y,
        N,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        C_IN=C_in,
        C_OUT=C_out,
        K_D=K_d,
        K_H=K_h,
        K_W=K_w,
        STRIDE_D=stride_d,
        STRIDE_H=stride_h,
        STRIDE_W=stride_w,
        PAD_D=pad_d,
        PAD_H=pad_h,
        PAD_W=pad_w,
        GROUPS=groups,
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_S=BLOCK_S,
    )

    return y


class ModelNew(nn.Module):
    """
    3D transposed convolution implemented with a high-performance Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the cubic convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side
            of each dimension in the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias. Defaults to False.
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
        assert (
            in_channels % groups == 0 and out_channels % groups == 0
        ), "in_channels and out_channels must be divisible by groups"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        # Normalize hyper-parameters
        self.stride = (stride, stride, stride)
        self.padding = (padding, padding, padding)
        self.output_padding = (output_padding, output_padding, output_padding)

        # Weight layout matches nn.ConvTranspose3d:
        # (in_channels, out_channels // groups, kD, kH, kW)
        self.weight = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels // groups,
                kernel_size,
                kernel_size,
                kernel_size,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        # Match nn.ConvTranspose3d initialization (Kaiming uniform)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
        )
