import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv3d_fwd_kernel(
    x_ptr,  # float32[N, Cin, D, H, W]
    w_ptr,  # float32[Cout, K] with K = Cin_per_group * KD * KH * KW
    b_ptr,  # float32[Cout] or dummy if HAS_BIAS=False
    y_ptr,  # float32[N, Cout, D_out, H_out, W_out]
    #
    N,
    Cin,
    D,
    H,
    W,
    Cout,
    D_out,
    H_out,
    W_out,
    Cin_per_group,
    Cout_per_group,
    groups,
    tiles_per_group,
    stride_d,
    stride_h,
    stride_w,
    pad_d,
    pad_h,
    pad_w,
    dil_d,
    dil_h,
    dil_w,
    x_stride_n,
    x_stride_c,
    x_stride_d,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_d,
    y_stride_h,
    y_stride_w,
    #
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # program ids
    pid_pos = tl.program_id(axis=0)
    pid_oc_all = tl.program_id(axis=1)

    # total number of output positions
    total_pos = N * D_out * H_out * W_out
    pos = pid_pos
    pos_mask = pos < total_pos

    # Decode pos -> (n, d_out, h_out, w_out)
    tmp = pos
    w_out_idx = tmp % W_out
    tmp = tmp // W_out
    h_out_idx = tmp % H_out
    tmp = tmp // H_out
    d_out_idx = tmp % D_out
    n_idx = tmp // D_out

    # Map pid_oc_all -> (group_id, tile_id_within_group)
    group_id = pid_oc_all // tiles_per_group
    tile_in_group = pid_oc_all % tiles_per_group

    oc_start = group_id * Cout_per_group + tile_in_group * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < Cout
    valid_oc = oc_mask & pos_mask

    # Initialize accumulator
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Precompute input base coordinates for this output position
    in_z_base = d_out_idx * stride_d - pad_d
    in_y_base = h_out_idx * stride_h - pad_h
    in_x_base = w_out_idx * stride_w - pad_w

    # Reduction over K dimension (Cin_per_group * KD * KH * KW)
    KDHW = KD * KH * KW
    KHW = KH * KW

    for k_block_start in range(0, K, BLOCK_K):
        k_idx = k_block_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Decode k_idx -> (cin_inner, kd, kh, kw)
        cin_inner = k_idx // KDHW
        rem = k_idx % KDHW
        kd_idx = rem // KHW
        rem2 = rem % KHW
        kh_idx = rem2 // KW
        kw_idx = rem2 % KW

        # Input channel index (taking groups into account)
        in_c = group_id * Cin_per_group + cin_inner

        # Spatial positions in input
        in_z = in_z_base + kd_idx * dil_d
        in_y = in_y_base + kh_idx * dil_h
        in_x = in_x_base + kw_idx * dil_w

        # Bounds check for input
        in_bounds = (
            (in_c >= 0)
            & (in_c < Cin)
            & (in_z >= 0)
            & (in_z < D)
            & (in_y >= 0)
            & (in_y < H)
            & (in_x >= 0)
            & (in_x < W)
        )
        load_mask_x = k_mask & in_bounds & pos_mask

        # Compute input pointer offsets
        x_offsets = (
            n_idx * x_stride_n
            + in_c * x_stride_c
            + in_z * x_stride_d
            + in_y * x_stride_h
            + in_x * x_stride_w
        )

        x_vals = tl.load(x_ptr + x_offsets, mask=load_mask_x, other=0.0)

        # Load weight block: [BLOCK_OC, BLOCK_K]
        oc_broadcast = oc_offsets[:, None]
        k_broadcast = k_idx[None, :]
        w_offsets = oc_broadcast * K + k_broadcast
        w_mask = (oc_broadcast < Cout) & k_mask[None, :] & pos_mask
        w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

        # Matrix-vector multiply via tl.dot
        # w_vals: [BLOCK_OC, BLOCK_K], x_vals: [BLOCK_K]
        partial = tl.dot(w_vals, x_vals[:, None], allow_tf32=True)  # [BLOCK_OC, 1]
        acc += partial[:, 0]

    # Add bias if present
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + oc_offsets, mask=valid_oc, other=0.0)
        acc += bias_vals

    # Store to output
    y_offsets = (
        n_idx * y_stride_n
        + oc_offsets * y_stride_c
        + d_out_idx * y_stride_d
        + h_out_idx * y_stride_h
        + w_out_idx * y_stride_w
    )

    tl.store(y_ptr + y_offsets, acc, mask=valid_oc)


def triton_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride,
    padding,
    dilation,
    groups: int,
) -> torch.Tensor:
    """
    High-performance 3D convolution using Triton.

    Args:
        x: Input tensor of shape [N, Cin, D, H, W], float32, CUDA.
        weight: [Cout, Cin/groups, KD, KH, KW], float32, CUDA.
        bias: [Cout] or None.
        stride, padding, dilation: int or 3-tuple.
        groups: number of groups.

    Returns:
        y: [N, Cout, D_out, H_out, W_out], float32, CUDA.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, "Only float32 supported"
    assert x.ndim == 5 and weight.ndim == 5, "x must be [N,C,D,H,W], weight [Cout,Cin/groups,KD,KH,KW]"

    N, Cin, D, H, W = x.shape
    Cout, Cin_per_group, KD, KH, KW = weight.shape
    assert Cin == Cin_per_group * groups, "Cin must equal Cin_per_group * groups"
    if bias is not None:
        assert bias.shape[0] == Cout

    # Normalize stride/padding/dilation to 3-tuples
    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_d = pad_h = pad_w = padding
    else:
        pad_d, pad_h, pad_w = padding
    if isinstance(dilation, int):
        dil_d = dil_h = dil_w = dilation
    else:
        dil_d, dil_h, dil_w = dilation

    # Output dimensions (match PyTorch Conv3d)
    D_out = (D + 2 * pad_d - dil_d * (KD - 1) - 1) // stride_d + 1
    H_out = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    # Make contiguous tensors
    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    # Flatten weight to [Cout, K]
    K = Cin_per_group * KD * KH * KW
    w_flat = w_contig.view(Cout, K)
    if bias is not None:
        b_contig = bias.contiguous()
    else:
        # Dummy tensor; won't be used when HAS_BIAS=False
        b_contig = torch.empty(1, device=x.device, dtype=x.dtype)

    y = torch.empty((N, Cout, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides
    x_s = x_contig.stride()
    y_s = y.stride()

    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w = x_s
    y_stride_n, y_stride_c, y_stride_d, y_stride_h, y_stride_w = y_s

    # Tiling parameters (power-of-two as required)
    BLOCK_OC = 64
    BLOCK_K = 32

    Cout_per_group = Cout // groups
    tiles_per_group = triton.cdiv(Cout_per_group, BLOCK_OC)

    total_pos = N * D_out * H_out * W_out
    grid = lambda meta: (
        max(1, total_pos),  # ensure >0
        tiles_per_group * groups,
    )

    conv3d_fwd_kernel[grid](
        x_contig,
        w_flat,
        b_contig,
        y,
        #
        N,
        Cin,
        D,
        H,
        W,
        Cout,
        D_out,
        H_out,
        W_out,
        Cin_per_group,
        Cout_per_group,
        groups,
        tiles_per_group,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dil_d,
        dil_h,
        dil_w,
        x_stride_n,
        x_stride_c,
        x_stride_d,
        x_stride_h,
        x_stride_w,
        y_stride_n,
        y_stride_c,
        y_stride_d,
        y_stride_h,
        y_stride_w,
        #
        KD=KD,
        KH=KH,
        KW=KW,
        K=K,
        BLOCK_OC=BLOCK_OC,
        BLOCK_K=BLOCK_K,
        HAS_BIAS=bias is not None,
    )

    return y


class ModelNew(nn.Module):
    """
    3D convolution module using a high-performance Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel (KD=KH=KW).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        # Use nn.Conv3d only as parameter container (weights + bias initialization)
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
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
        """
        Performs the 3D convolution using Triton.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        return triton_conv3d(
            x,
            self.conv3d.weight,
            self.conv3d.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
