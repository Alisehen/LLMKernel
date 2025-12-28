# <complete ModelNew code with optimized Triton kernels>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,  # (N, C_in, D_in, H_in, W_in)
    w_ptr,  # (C_in, C_out_per_group, K_D, K_H, K_W)
    b_ptr,  # (C_out,) or dummy if HAS_BIAS=False
    y_ptr,  # (N, C_out, D_out, H_out, W_out)
    N, C_in, D_in, H_in, W_in,
    C_out, D_out, H_out, W_out,
    total_pos,          # N * D_out * H_out * W_out
    pos_stride_y,       # how many positions are mapped along grid axis-1
    C_OUT_PER_GROUP: tl.constexpr,
    C_IN_PER_GROUP: tl.constexpr,
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
    NUM_CO_BLOCKS_PER_GROUP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    # ----------------------------------------------------------------------
    # Program IDs
    # ----------------------------------------------------------------------
    pid_co = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)

    # Flattened output position this program handles
    pos = pid_z * pos_stride_y + pid_y
    if pos >= total_pos:
        # Safe early-exit (not inside a loop, so allowed)
        return

    # Map pid_co -> (group_id, block_in_group)
    group_id = pid_co // NUM_CO_BLOCKS_PER_GROUP
    block_in_group = pid_co % NUM_CO_BLOCKS_PER_GROUP

    # Offsets for output channels this program computes
    co_offsets = block_in_group * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_offsets = co_offsets + group_id * C_OUT_PER_GROUP
    mask_co = co_offsets < C_out

    # Decode flattened output position -> (n, d_out, h_out, w_out)
    W_out_i = W_out
    H_out_i = H_out
    D_out_i = D_out

    w_out = pos % W_out_i
    tmp = pos // W_out_i
    h_out = tmp % H_out_i
    tmp = tmp // H_out_i
    d_out = tmp % D_out_i
    n = tmp // D_out_i

    # Accumulator for BLOCK_CO output channels
    acc = tl.zeros([BLOCK_CO], dtype=tl.float32)

    # Optional bias
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0)
        acc += bias_vals

    # Local co index within group and kernel volume
    co_local_offsets = co_offsets - group_id * C_OUT_PER_GROUP
    K_ELEMS = K_D * K_H * K_W

    # Main transposed-convolution accumulation
    for ci_local in range(C_IN_PER_GROUP):
        c_in_global = group_id * C_IN_PER_GROUP + ci_local

        for kd in range(K_D):
            num_d = d_out + PAD_D - kd
            mask_d = num_d >= 0
            mask_d = mask_d & (num_d < STRIDE_D * D_in)
            mask_d = mask_d & (num_d % STRIDE_D == 0)
            d_in = num_d // STRIDE_D
            mask_d = mask_d & (d_in >= 0) & (d_in < D_in)

            for kh in range(K_H):
                num_h = h_out + PAD_H - kh
                mask_h = num_h >= 0
                mask_h = mask_h & (num_h < STRIDE_H * H_in)
                mask_h = mask_h & (num_h % STRIDE_H == 0)
                h_in = num_h // STRIDE_H
                mask_h = mask_h & (h_in >= 0) & (h_in < H_in)

                for kw in range(K_W):
                    num_w = w_out + PAD_W - kw
                    mask_w = num_w >= 0
                    mask_w = mask_w & (num_w < STRIDE_W * W_in)
                    mask_w = mask_w & (num_w % STRIDE_W == 0)
                    w_in = num_w // STRIDE_W
                    mask_w = mask_w & (w_in >= 0) & (w_in < W_in)

                    mask_in = mask_d & mask_h & mask_w

                    # Input index (scalar)
                    x_index = (
                        ((n * C_in + c_in_global) * D_in + d_in) * H_in + h_in
                    ) * W_in + w_in

                    x_val = tl.load(x_ptr + x_index, mask=mask_in, other=0.0)

                    # Weight base index for co_local = 0
                    w_base = (
                        ((c_in_global * C_OUT_PER_GROUP) * K_D + kd) * K_H + kh
                    ) * K_W + kw

                    # Vector of weight indices over co_local
                    w_offsets = w_base + co_local_offsets * K_ELEMS
                    w_vals = tl.load(w_ptr + w_offsets, mask=mask_co, other=0.0)

                    acc += x_val * w_vals

    # Store results
    y_offsets = (
        ((n * C_out + co_offsets) * D_out + d_out) * H_out + h_out
    ) * W_out + w_out
    tl.store(y_ptr + y_offsets, acc, mask=mask_co)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
    output_padding: int,
    groups: int,
) -> torch.Tensor:
    # Ensure contiguity
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_per_group, K_D, K_H, K_W = weight.shape
    assert C_in_w == C_in
    assert K_D == K_H == K_W

    C_out = C_out_per_group * groups

    stride_d = stride_h = stride_w = stride
    pad_d = pad_h = pad_w = padding

    D_out = (D_in - 1) * stride_d - 2 * pad_d + K_D + output_padding
    H_out = (H_in - 1) * stride_h - 2 * pad_h + K_H + output_padding
    W_out = (W_in - 1) * stride_w - 2 * pad_w + K_W + output_padding

    # Allocate output
    y = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=x.device,
        dtype=x.dtype,
    )

    # Early-exit for degenerate cases to avoid 0-sized grid launches
    if (
        N == 0
        or C_out == 0
        or D_out <= 0
        or H_out <= 0
        or W_out <= 0
    ):
        return y

    BLOCK_CO = 32  # power-of-2 as required

    C_OUT_PER_GROUP = C_out_per_group
    C_IN_PER_GROUP = C_in // groups

    total_pos = N * D_out * H_out * W_out
    NUM_CO_BLOCKS_PER_GROUP = triton.cdiv(C_OUT_PER_GROUP, BLOCK_CO)

    # Respect CUDA grid limits on y/z (typically 65535)
    MAX_GRID_AXIS_YZ = 65535
    pos_stride_y = min(total_pos, MAX_GRID_AXIS_YZ)
    grid_y = pos_stride_y
    grid_z = triton.cdiv(total_pos, grid_y)

    grid = (
        NUM_CO_BLOCKS_PER_GROUP * groups,
        grid_y,
        grid_z,
    )

    conv_transpose3d_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,  # dummy when HAS_BIAS=False
        y,
        N,
        C_in,
        D_in,
        H_in,
        W_in,
        C_out,
        D_out,
        H_out,
        W_out,
        total_pos,
        pos_stride_y,
        C_OUT_PER_GROUP=C_OUT_PER_GROUP,
        C_IN_PER_GROUP=C_IN_PER_GROUP,
        K_D=K_D,
        K_H=K_H,
        K_W=K_W,
        STRIDE_D=stride_d,
        STRIDE_H=stride_h,
        STRIDE_W=stride_w,
        PAD_D=pad_d,
        PAD_H=pad_h,
        PAD_W=pad_w,
        GROUPS=groups,
        NUM_CO_BLOCKS_PER_GROUP=NUM_CO_BLOCKS_PER_GROUP,
        HAS_BIAS=bias is not None,
        BLOCK_CO=BLOCK_CO,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for a 3D transposed convolution with
    square kernel, isotropic stride/padding, and groups.
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

        # Initialize weights using PyTorch's ConvTranspose3d initializer
        ref = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
        self.weight = nn.Parameter(ref.weight.detach())
        if bias:
            self.bias = nn.Parameter(ref.bias.detach())
        else:
            self.bias = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

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
