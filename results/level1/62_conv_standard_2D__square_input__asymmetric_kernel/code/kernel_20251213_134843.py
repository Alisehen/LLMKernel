import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_CO': 64, 'BLOCK_HW': 64, 'BLOCK_CI': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_CO': 128, 'BLOCK_HW': 32, 'BLOCK_CI': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_CO': 64, 'BLOCK_HW': 128, 'BLOCK_CI': 32}, num_warps=8, num_stages=3),
    ],
    key=['COUT_PER_GROUP', 'H_OUT', 'W_OUT'],
)
@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_IN, H_IN, W_IN,
    C_OUT, K_H, K_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    DIL_H, DIL_W,
    GROUPS,
    H_OUT, W_OUT,
    x_batch_stride, x_channel_stride, x_h_stride, x_w_stride,
    w_out_stride, w_c_stride, w_h_stride, w_w_stride,
    y_batch_stride, y_channel_stride, y_h_stride, y_w_stride,
    CIN_PER_GROUP,
    COUT_PER_GROUP,
    BLOCKS_PER_GROUP,
    Y_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid_co = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_n = tl.program_id(2)

    group_id = pid_co // BLOCKS_PER_GROUP
    block_in_group = pid_co % BLOCKS_PER_GROUP

    oc_start = group_id * COUT_PER_GROUP + block_in_group * BLOCK_CO
    oc_idx = oc_start + tl.arange(0, BLOCK_CO)
    oc_mask = (oc_idx < (group_id + 1) * COUT_PER_GROUP) & (oc_idx < C_OUT)

    total_hw = H_OUT * W_OUT
    hw_start = pid_hw * BLOCK_HW
    hw_idx = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_idx < total_hw

    oh = hw_idx // W_OUT
    ow = hw_idx % W_OUT

    base_h = oh * STRIDE_H - PAD_H
    base_w = ow * STRIDE_W - PAD_W

    cin_base = group_id * CIN_PER_GROUP

    acc = tl.zeros((BLOCK_CO, BLOCK_HW), dtype=tl.float32)

    ci_range = tl.arange(0, BLOCK_CI)

    for ci_start in range(0, CIN_PER_GROUP, BLOCK_CI):
        ci_idx = ci_start + ci_range
        ci_mask = ci_idx < CIN_PER_GROUP
        chan_ids = cin_base + ci_idx

        for kh in range(0, K_H):
            ih = base_h + kh * DIL_H
            h_valid = (ih >= 0) & (ih < H_IN)

            for kw in range(0, K_W):
                iw = base_w + kw * DIL_W
                w_valid = (iw >= 0) & (iw < W_IN)
                spatial_mask = hw_mask & h_valid & w_valid

                x_ptrs = (
                    x_ptr
                    + pid_n * x_batch_stride
                    + chan_ids[:, None] * x_channel_stride
                    + ih[None, :] * x_h_stride
                    + iw[None, :] * x_w_stride
                )
                load_mask = ci_mask[:, None] & spatial_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=load_mask, other=0.0).to(tl.float32)

                w_ptrs = (
                    w_ptr
                    + oc_idx[:, None] * w_out_stride
                    + ci_idx[None, :] * w_c_stride
                    + kh * w_h_stride
                    + kw * w_w_stride
                )
                w_mask = oc_mask[:, None] & ci_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

                acc += tl.dot(w_vals, x_vals)

    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + oc_idx, mask=oc_mask, other=0.0).to(tl.float32)
        acc += bias_vals[:, None]

    out = acc.to(Y_DTYPE)
    y_ptrs = (
        y_ptr
        + pid_n * y_batch_stride
        + oc_idx[:, None] * y_channel_stride
        + oh[None, :] * y_h_stride
        + ow[None, :] * y_w_stride
    )
    out_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(y_ptrs, out, mask=out_mask)


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.bfloat16:
        return tl.bfloat16
    raise ValueError(f"Unsupported dtype for Triton convolution: {dtype}")


def conv2d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
):
    N, C_in, H_in, W_in = x.shape
    C_out, _, K_h, K_w = weight.shape

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    H_out = (H_in + 2 * pad_h - dil_h * (K_h - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dil_w * (K_w - 1) - 1) // stride_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_CO = 64
    BLOCK_HW = 64
    BLOCK_CI = 32

    cin_per_group = C_in // groups
    cout_per_group = C_out // groups
    blocks_per_group = (cout_per_group + BLOCK_CO - 1) // BLOCK_CO

    grid = (
        groups * blocks_per_group,
        triton.cdiv(H_out * W_out, BLOCK_HW),
        N,
    )

    Y_DTYPE = _torch_dtype_to_triton(y.dtype)

    conv2d_nchw_kernel[grid](
        x,
        weight,
        bias if bias is not None else y,
        y,
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        K_h,
        K_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        groups,
        H_out,
        W_out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        cin_per_group,
        cout_per_group,
        blocks_per_group,
        Y_DTYPE=Y_DTYPE,
        HAS_BIAS=bias is not None,
        BLOCK_CO=BLOCK_CO,
        BLOCK_HW=BLOCK_HW,
        BLOCK_CI=BLOCK_CI,
    )
    return y


class ModelNew(nn.Module):
    """
    High-performance Triton-optimized 2D convolution module supporting asymmetric kernels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        k_h, k_w = kernel_size
        cin_per_group = in_channels // groups
        weight = torch.empty(
            (out_channels, cin_per_group, k_h, k_w), dtype=torch.float32
        )
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        if bias:
            fan_in = cin_per_group * k_h * k_w
            bound = 1 / math.sqrt(fan_in)
            bias_tensor = torch.empty(out_channels, dtype=torch.float32).uniform_(-bound, bound)
            self.bias = nn.Parameter(bias_tensor)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if x.dtype != weight.dtype:
            weight = weight.to(x.dtype)
            if bias is not None:
                bias = bias.to(x.dtype)

        return conv2d_triton(
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
