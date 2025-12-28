import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr,              # *f32 [N, C_in, D_in, H_in, W_in]
    w_ptr,              # *f32 [C_in, C_out_group, K, K, K]
    b_ptr,              # *f32 [C_out] or dummy if no bias
    y_ptr,              # *f32 [N, C_out, D_out, H_out, W_out]
    N,                  # int
    D_IN, H_IN, W_IN,   # int
    D_OUT, H_OUT, W_OUT,# int
    n_elements,         # int = N * C_out * D_out * H_out * W_OUT
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    K: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    GROUPS: tl.constexpr,
    C_IN_GROUP: tl.constexpr,
    C_OUT_GROUP: tl.constexpr,
    USE_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Decode linear index into (n, co, od, oh, ow)
    tmp = offs
    ow = tmp % W_OUT
    tmp = tmp // W_OUT
    oh = tmp % H_OUT
    tmp = tmp // H_OUT
    od = tmp % D_OUT
    tmp = tmp // D_OUT
    co = tmp % C_OUT
    tmp = tmp // C_OUT
    n = tmp  # [0, N)

    group_id = co // C_OUT_GROUP
    c_out_group = co % C_OUT_GROUP

    # Accumulator in fp32 for better numerical stability
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over input channels in this group and kernel volume
    for ci_g in range(C_IN_GROUP):
        c_in = group_id * C_IN_GROUP + ci_g

        # Base weight offset for (c_in, c_out_group, :, :, :)
        w_ci_co_base = (c_in * C_OUT_GROUP + c_out_group) * (K * K * K)

        for kd in range(K):
            tmp_d = od + PADDING - kd
            id_ = tmp_d // STRIDE
            mask_d = (tmp_d >= 0) & (tmp_d % STRIDE == 0) & (id_ < D_IN)

            for kh in range(K):
                tmp_h = oh + PADDING - kh
                ih = tmp_h // STRIDE
                mask_h = (tmp_h >= 0) & (tmp_h % STRIDE == 0) & (ih < H_IN)

                mask_dh = mask & mask_d & mask_h

                for kw in range(K):
                    tmp_w = ow + PADDING - kw
                    iw = tmp_w // STRIDE
                    mask_w = (tmp_w >= 0) & (tmp_w % STRIDE == 0) & (iw < W_IN)

                    final_mask = mask_dh & mask_w

                    # Input index: (n, c_in, id_, ih, iw)
                    x_idx = (
                        (((n * C_IN) + c_in) * D_IN + id_) * H_IN + ih
                    ) * W_IN + iw
                    x_val = tl.load(x_ptr + x_idx, mask=final_mask, other=0.0)

                    # Weight index: (c_in, c_out_group, kd, kh, kw)
                    w_k_offset = ((kd * K) + kh) * K + kw
                    w_idx = w_ci_co_base + w_k_offset
                    w_val = tl.load(w_ptr + w_idx)

                    acc += x_val * w_val

    # Add bias if present
    if USE_BIAS:
        b_val = tl.load(b_ptr + co, mask=mask, other=0.0)
        acc += b_val

    # Store result
    tl.store(y_ptr + offs, acc, mask=mask)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    padding: int,
    output_padding: int,
    groups: int,
) -> torch.Tensor:
    """
    Triton-accelerated ConvTranspose3d for cubic kernels with equal stride/padding.
    Uses Triton on CUDA tensors (supports arbitrary output_padding).
    Falls back to torch.nn.functional.conv_transpose3d on non-CUDA tensors.
    """
    use_triton = (
        x.is_cuda
        and weight.is_cuda
        and (bias is None or bias.is_cuda)
    )

    if not use_triton:
        # CPU or non-CUDA tensors: rely on PyTorch implementation
        return torch.nn.functional.conv_transpose3d(
            x,
            weight,
            bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )

    # Triton path
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out_group, Kd, Kh, Kw = weight.shape
    assert C_in_w == C_in, "Weight in_channels must match input channels"
    assert Kd == Kh == Kw, "Kernel must be cubic"
    K = Kd
    C_out = C_out_group * groups

    # Output shape (dilation = 1)
    D_out = (D_in - 1) * stride - 2 * padding + K + output_padding
    H_out = (H_in - 1) * stride - 2 * padding + K + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K + output_padding

    y = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    n_elements = y.numel()
    BLOCK_SIZE = 256  # power of 2

    grid = lambda meta: (max(1, triton.cdiv(n_elements, meta["BLOCK_SIZE"])),)

    conv_transpose3d_kernel[grid](
        x,                                      # x_ptr
        weight,                                # w_ptr
        bias if bias is not None else x,       # b_ptr (dummy if no bias)
        y,                                     # y_ptr
        N,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        n_elements,
        STRIDE=stride,
        PADDING=padding,
        K=K,
        C_IN=C_in,
        C_OUT=C_out,
        GROUPS=groups,
        C_IN_GROUP=C_in // groups,
        C_OUT_GROUP=C_out // groups,
        USE_BIAS=(bias is not None),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y


class TritonConvTranspose3d(nn.Module):
    """
    ConvTranspose3d implemented with a Triton kernel on CUDA tensors.
    Matches nn.ConvTranspose3d interface for cubic kernels with equal stride/padding.
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
        if isinstance(kernel_size, tuple):
            assert (
                len(kernel_size) == 3
                and kernel_size[0] == kernel_size[1] == kernel_size[2]
            ), "Only cubic kernel_size is supported"
            k = kernel_size[0]
        else:
            k = kernel_size

        assert out_channels % groups == 0
        assert in_channels % groups == 0

        # For simplicity and performance we currently support equal stride/padding
        if isinstance(stride, tuple):
            assert len(stride) == 3 and stride[0] == stride[1] == stride[2], \
                "Triton path supports only equal stride in all dims"
            stride_val = stride[0]
        else:
            stride_val = stride

        if isinstance(padding, tuple):
            assert len(padding) == 3 and padding[0] == padding[1] == padding[2], \
                "Triton path supports only equal padding in all dims"
            padding_val = padding[0]
        else:
            padding_val = padding

        if isinstance(output_padding, tuple):
            assert len(output_padding) == 3 and \
                   output_padding[0] == output_padding[1] == output_padding[2], \
                "Triton path supports only equal output_padding in all dims"
            output_padding_val = output_padding[0]
        else:
            output_padding_val = output_padding

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.stride = stride_val
        self.padding = padding_val
        self.output_padding = output_padding_val
        self.groups = groups

        weight_shape = (in_channels, out_channels // groups, k, k, k)
        self.weight = nn.Parameter(torch.empty(weight_shape))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size * self.kernel_size
            bound = 1.0 / math.sqrt(fan_in)
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


class ModelNew(nn.Module):
    """
    Drop-in replacement model using TritonConvTranspose3d.
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
    ) -> None:
        super().__init__()
        self.conv_transpose3d = TritonConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose3d(x)
