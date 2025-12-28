import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_direct_fwd_kernel(
    x_ptr,        # *f32, [N, Ci, Di, Hi, Wi] contiguous
    w_ptr,        # *f32, [Co, Ci, Kd, Kh, Kw] contiguous
    b_ptr,        # *f32, [Co] (ignored if has_bias == False)
    out_ptr,      # *f32, [N, Co, Do, Ho, Wo] contiguous
    N,            # int
    Di, Hi, Wi,   # int
    Co,           # int
    Do, Ho, Wo,   # int
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    Ci: tl.constexpr,   # in_channels (compile-time for unrolling)
    Kd: tl.constexpr,
    Kh: tl.constexpr,
    Kw: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Direct 3D convolution kernel specialized for groups=1 and fp32.
    Each program instance computes a tile:
        - fixed (n, d_out, h_out)
        - BLOCK_OC contiguous output channels
        - BLOCK_W contiguous output width positions
    """
    # Program ids
    pid_n_dh = tl.program_id(axis=0)
    pid_oc = tl.program_id(axis=1)
    pid_w = tl.program_id(axis=2)

    # Decode (n, d_out, h_out) from pid_n_dh in [0, N*Do*Ho)
    tmp = pid_n_dh
    n_idx = tmp // (Do * Ho)
    tmp = tmp % (Do * Ho)
    d_out = tmp // Ho
    h_out = tmp % Ho

    # Offsets for output channels and width
    oc_start = pid_oc * BLOCK_OC
    w_start = pid_w * BLOCK_W

    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    w_offsets = w_start + tl.arange(0, BLOCK_W)

    mask_oc = oc_offsets < Co
    mask_w = w_offsets < Wo

    # Initialize accumulator [BLOCK_OC, BLOCK_W]
    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)

    # Precompute strides for weight indexing
    w_oc_stride = Ci * Kd * Kh * Kw  # number of elements per output channel

    # Base expression for input x indices: [N, Ci, Di, Hi, Wi]
    # We'll build it incrementally inside the loops to reduce integer ops.

    # Loop over input channels and kernel spatial dimensions
    for ci in range(Ci):
        # Base offset for this (n, ci) pair along depth: ((n * Ci + ci) * Di)
        base_n_ci = (n_idx * Ci + ci) * Di

        # Precompute per-ci term for weight indexing: ci * Kd * Kh * Kw
        w_ci_base = ci * (Kd * Kh * Kw)

        for kd in range(Kd):
            z_in = d_out * stride_d - pad_d + kd * dil_d
            valid_z = (z_in >= 0) & (z_in < Di)

            # Depth contribution to input index: ((base_n_ci + z_in) * Hi)
            base_n_ci_kd = (base_n_ci + z_in) * Hi

            # Weight base for this ci, kd: w_ci_base + kd * Kh * Kw
            w_ci_kd_base = w_ci_base + kd * (Kh * Kw)

            for kh in range(Kh):
                y_in = h_out * stride_h - pad_h + kh * dil_h
                valid_y = (y_in >= 0) & (y_in < Hi)

                # Height contribution to input index: ((base_n_ci_kd + y_in) * Wi)
                base_input_row = (base_n_ci_kd + y_in) * Wi

                # Weight base for this ci, kd, kh: w_ci_kd_base + kh * Kw
                w_ci_kd_kh_base = w_ci_kd_base + kh * Kw

                # Scalar mask for (z,y) being valid
                zy_valid = valid_z & valid_y

                for kw in range(Kw):
                    # Input x positions (vector over BLOCK_W)
                    x_in = w_offsets * stride_w - pad_w + kw * dil_w
                    mask_x = (x_in >= 0) & (x_in < Wi)

                    # Final mask for loading x
                    load_mask = zy_valid & mask_x & mask_w

                    # Compute input pointers
                    x_ptrs = x_ptr + base_input_row + x_in
                    x_vals = tl.load(x_ptrs, mask=load_mask, other=0.0)

                    # Weight index for this (ci, kd, kh, kw) and each oc
                    # index = oc * (Ci*Kd*Kh*Kw) + (ci,kd,kh,kw-flattened)
                    w_idx = oc_offsets * w_oc_stride + (w_ci_kd_kh_base + kw)
                    w_vals = tl.load(w_ptr + w_idx, mask=mask_oc, other=0.0)

                    # Outer product: [BLOCK_OC, 1] * [1, BLOCK_W]
                    acc += w_vals[:, None] * x_vals[None, :]

    # Add bias if present: broadcast along width
    if has_bias:
        bias_vals = tl.load(b_ptr + oc_offsets, mask=mask_oc, other=0.0)
        acc += bias_vals[:, None]

    # Write back to output: [N, Co, Do, Ho, Wo]
    # Compute flattened output indices
    w_idx = w_offsets[None, :]
    oc_idx = oc_offsets[:, None]

    out_index = (
        (((n_idx * Co + oc_idx) * Do + d_out) * Ho + h_out) * Wo + w_idx
    )

    out_mask = mask_oc[:, None] & mask_w[None, :]
    tl.store(out_ptr + out_index, acc, mask=out_mask)


def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
) -> torch.Tensor:
    # Fallbacks for unsupported cases
    if (not x.is_cuda) or (not weight.is_cuda):
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
    if groups != 1:
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

    # Ensure contiguous
    x = x.contiguous()
    weight = weight.contiguous()

    N, Ci, Di, Hi, Wi = x.shape
    Co, Ci_w, Kd, Kh, Kw = weight.shape
    assert Ci_w == Ci, "Incompatible in_channels between input and weight for groups=1."

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation

    # Output spatial dimensions (PyTorch formula)
    Do = (Di + 2 * pad_d - dil_d * (Kd - 1) - 1) // stride_d + 1
    Ho = (Hi + 2 * pad_h - dil_h * (Kh - 1) - 1) // stride_h + 1
    Wo = (Wi + 2 * pad_w - dil_w * (Kw - 1) - 1) // stride_w + 1

    # Degenerate / invalid shapes fall back to PyTorch
    if Do <= 0 or Ho <= 0 or Wo <= 0:
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

    out = torch.empty((N, Co, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    K_total = Ci * Kd * Kh * Kw
    if N == 0 or Co == 0 or K_total == 0 or Do == 0 or Ho == 0 or Wo == 0:
        return torch.nn.functional.conv3d(
            x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups
        )

    # Tile sizes (power-of-2 as required)
    BLOCK_OC = 32
    BLOCK_W = 32

    # Grid:
    #  axis0: one program per (n, d_out, h_out)
    #  axis1: tiles of output channels
    #  axis2: tiles of output width
    grid = lambda meta: (
        N * Do * Ho,
        triton.cdiv(Co, meta["BLOCK_OC"]),
        triton.cdiv(Wo, meta["BLOCK_W"]),
    )

    has_bias = bias is not None
    bias_ptr = bias if bias is not None else weight  # dummy pointer when has_bias == False

    conv3d_direct_fwd_kernel[grid](
        x,
        weight,
        bias_ptr,
        out,
        N,
        Di,
        Hi,
        Wi,
        Co,
        Do,
        Ho,
        Wo,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w,
        Ci=Ci,
        Kd=Kd,
        Kh=Kh,
        Kw=Kw,
        has_bias=has_bias,
        BLOCK_OC=BLOCK_OC,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )

    return out


class ModelNew(nn.Module):
    """
    3D convolution implemented with a high-performance Triton kernel (groups=1 fast path).
    Falls back to PyTorch's conv3d for non-CUDA tensors, non-fp32 dtypes, or groups != 1.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        dilation: tuple = (1, 1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        # Use nn.Conv3d only for parameter management / initialization
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            x,
            self.conv3d.weight,
            self.conv3d.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
