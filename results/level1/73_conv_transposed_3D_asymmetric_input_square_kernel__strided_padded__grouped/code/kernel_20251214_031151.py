# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    input_ptr,        # *f32:  [N, C_in, ID, IH, IW]
    weight_ptr,       # *f32:  [C_in, C_out_per_group, K, K, K]
    bias_ptr,         # *f32:  [C_out]  (or dummy if HAS_BIAS == False)
    output_ptr,       # *f32:  [N, C_out, OD, OH, OW]
    N, C_in, C_out,
    ID, IH, IW,
    OD, OH, OW,
    C_in_per_group, C_out_per_group,
    stride, padding,
    total_out_elements,
    HAS_BIAS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    # Program IDs
    pid_out_flat = tl.program_id(axis=0)  # over N*OD*OH*OW
    pid_c_out_g = tl.program_id(axis=1)   # over C_out_per_group
    pid_group = tl.program_id(axis=2)     # over groups

    # Flat output indices this program handles
    out_start = pid_out_flat * BLOCK_OUT
    offsets = out_start + tl.arange(0, BLOCK_OUT)
    mask_o = offsets < total_out_elements

    # Decode (n, od, oh, ow) from flat offsets
    OD_OH_OW = OD * OH * OW
    OH_OW = OH * OW

    n = offsets // OD_OH_OW
    rem = offsets % OD_OH_OW
    od = rem // OH_OW
    rem = rem % OH_OW
    oh = rem // OW
    ow = rem % OW

    # Channel / group indexing
    c_out_in_group = pid_c_out_g
    group_id = pid_group

    c_out = group_id * C_out_per_group + c_out_in_group
    c_in_start = group_id * C_in_per_group

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

    # Loop over kernel positions
    for kd in range(KERNEL_SIZE):
        for kh in range(KERNEL_SIZE):
            for kw in range(KERNEL_SIZE):
                # Compute corresponding input coordinates
                nom_d = od + padding - kd
                id = nom_d // stride
                nom_h = oh + padding - kh
                ih = nom_h // stride
                nom_w = ow + padding - kw
                iw = nom_w // stride

                # Validity checks per dimension
                valid_d = (nom_d % stride == 0) & (id >= 0) & (id < ID)
                valid_h = (nom_h % stride == 0) & (ih >= 0) & (ih < IH)
                valid_w = (nom_w % stride == 0) & (iw >= 0) & (iw < IW)

                valid = valid_d & valid_h & valid_w & mask_o

                # Skip work if no lane is valid for this (kd, kh, kw)
                if tl.sum(valid) == 0:
                    # Triton executes this per program; allowed since 'valid' is a tl.tensor
                    # Note: this condition is on a scalar result of tl.sum, not on a tl.tensor.
                    pass

                # Loop over input channels in this group
                for c_in_offset in range(0, C_in_per_group):
                    c_in = c_in_start + c_in_offset

                    # Flat input index: (((n * C_in + c_in) * ID + id) * IH + ih) * IW + iw
                    in_index = (((n * C_in + c_in) * ID + id) * IH + ih) * IW + iw

                    x = tl.load(input_ptr + in_index, mask=valid, other=0.0)
                    x = x.to(tl.float32)

                    # Flat weight index: [C_in, C_out_per_group, K, K, K]
                    # idx = ((((c_in * C_out_per_group) + c_out_in_group) * K + kd) * K + kh) * K + kw
                    w_index = ((((c_in * C_out_per_group) + c_out_in_group)
                                * KERNEL_SIZE + kd)
                               * KERNEL_SIZE + kh) * KERNEL_SIZE + kw
                    w = tl.load(weight_ptr + w_index)
                    w = w.to(tl.float32)

                    acc += x * w

    # Add bias if present
    if HAS_BIAS:
        b = tl.load(bias_ptr + c_out)
        b = b.to(tl.float32)
        acc = acc + b

    # Write results
    out_index = (((n * C_out + c_out) * OD + od) * OH + oh) * OW + ow
    tl.store(output_ptr + out_index, acc, mask=mask_o)


def conv_transpose3d_triton(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            stride: int,
                            padding: int,
                            groups: int) -> torch.Tensor:
    """
    Triton implementation of 3D transposed convolution with:
      - cubic kernel (K, K, K)
      - uniform stride in all dimensions
      - uniform padding in all dimensions
      - arbitrary groups
      - output_padding = 0, dilation = 1
    Falls back to PyTorch on CPU.
    """
    # CPU fallback
    if not x.is_cuda:
        return F.conv_transpose3d(
            x, weight, bias,
            stride=stride,
            padding=padding,
            output_padding=0,
            groups=groups,
            dilation=1,
        )

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias_t = bias.contiguous()
    else:
        bias_t = None

    N, C_in, ID, IH, IW = x.shape
    C_in_w, C_out_per_group, Kd, Kh, Kw = weight.shape
    assert C_in_w == C_in, "Weight in_channels must match input"
    assert Kd == Kh == Kw, "Kernel must be cubic"
    kernel_size = Kd

    assert stride >= 1, "Stride must be >= 1"
    assert padding >= 0, "Padding must be >= 0"
    assert C_in % groups == 0, "in_channels must be divisible by groups"
    C_in_per_group = C_in // groups

    C_out = C_out_per_group * groups
    assert C_out > 0
    assert C_out % groups == 0
    C_out_per_group = C_out // groups

    # Output size: (ID - 1) * stride - 2 * padding + kernel_size
    OD = (ID - 1) * stride - 2 * padding + kernel_size
    OH = (IH - 1) * stride - 2 * padding + kernel_size
    OW = (IW - 1) * stride - 2 * padding + kernel_size

    output = torch.empty((N, C_out, OD, OH, OW),
                         device=x.device,
                         dtype=x.dtype)

    total_out_elements = N * OD * OH * OW

    BLOCK_OUT = 128  # power-of-two as required

    grid = (
        triton.cdiv(total_out_elements, BLOCK_OUT),
        C_out_per_group,
        groups,
    )

    has_bias = bias_t is not None
    # When HAS_BIAS == False, bias_ptr is never used; pass a dummy tensor.
    bias_ptr = bias_t if has_bias else output

    conv_transpose3d_kernel[grid](
        x, weight, bias_ptr, output,
        N, C_in, C_out,
        ID, IH, IW,
        OD, OH, OW,
        C_in_per_group, C_out_per_group,
        stride, padding,
        total_out_elements,
        HAS_BIAS=has_bias,
        KERNEL_SIZE=kernel_size,
        BLOCK_OUT=BLOCK_OUT,
    )

    return output


class ModelNew(nn.Module):
    """
    Same interface as the original Model, but uses a Triton kernel
    to implement ConvTranspose3d in the forward pass.
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
        # Keep the PyTorch module for parameter storage / initialization
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv_transpose3d
        # Use Triton kernel on CUDA tensors, otherwise fall back
        return conv_transpose3d_triton(
            x,
            conv.weight,
            conv.bias,
            stride=conv.stride[0],
            padding=conv.padding[0],
            groups=conv.groups,
        )
