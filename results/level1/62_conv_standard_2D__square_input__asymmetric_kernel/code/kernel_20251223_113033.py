import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_fwd_kernel(
    x_ptr,  # (N, C_in, H_in, W_in)
    w_ptr,  # (C_out, C_in/groups, KH, KW)
    b_ptr,  # (C_out) or unused if HAS_BIAS=False
    y_ptr,  # (N, C_out, H_out, W_out)
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    cin_per_group,
    cout_per_group,
    N_HW,  # N * H_out * W_out
    x_stride0, x_stride1, x_stride2, x_stride3,
    w_stride0, w_stride1, w_stride2, w_stride3,
    y_stride0, y_stride1, y_stride2, y_stride3,
    OC_BLOCKS_PER_GROUP,
    BLOCK_M: tl.constexpr,  # out_channels tile
    BLOCK_N: tl.constexpr,  # output-position tile
    BLOCK_K: tl.constexpr,  # reduction tile: cin_per_group * KH * KW
    HAS_BIAS: tl.constexpr,
):
    # Program ids
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    # Decode group and oc-block within group
    group_id = pid0 // OC_BLOCKS_PER_GROUP
    oc_block_id = pid0 % OC_BLOCKS_PER_GROUP

    # Offsets along output-channel (within group) and output-position dimensions
    offs_m = oc_block_id * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M], 0..cout_per_group-1 (within group)
    offs_n = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)         # [BLOCK_N], 0..N_HW-1

    # Global out-channel indices
    group_oc_start = group_id * cout_per_group
    oc_idx = group_oc_start + offs_m  # [BLOCK_M]

    # Masks
    mask_m = offs_m < cout_per_group
    mask_n = offs_n < N_HW

    # Compute (n, ho, wo) for each output index in offs_n
    HW_out = H_out * W_out
    n_idx = offs_n // HW_out
    rem_sp = offs_n % HW_out
    ho_idx = rem_sp // W_out
    wo_idx = rem_sp % W_out

    # Total reduction size
    K_total = cin_per_group * KH * KW

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K dimension
    for k_start in range(0, K_total, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = offs_k < K_total

        # Map K index -> (ci_g, kh, kw)
        khkw = KH * KW
        ci_g = offs_k // khkw
        rem_k = offs_k % khkw
        kh = rem_k // KW
        kw = rem_k % KW

        # --------- Load Weights W: shape (C_out, cin_per_group, KH, KW) ----------
        oc_bcast = oc_idx[:, None]           # [BLOCK_M, 1]
        ci_bcast = ci_g[None, :]            # [1, BLOCK_K]
        kh_bcast_w = kh[None, :]            # [1, BLOCK_K]
        kw_bcast_w = kw[None, :]            # [1, BLOCK_K]

        w_ptrs = (
            w_ptr
            + oc_bcast * w_stride0
            + ci_bcast * w_stride1
            + kh_bcast_w * w_stride2
            + kw_bcast_w * w_stride3
        )
        mask_w = (mask_m[:, None]) & (mask_k[None, :])

        w_vals = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # --------- Load Input X as implicit im2col: B matrix (K, N) ----------
        ic_full = group_id * cin_per_group + ci_g  # [BLOCK_K]

        ic_bcast = ic_full[:, None]       # [BLOCK_K, 1]
        n_bcast = n_idx[None, :]          # [1, BLOCK_N]
        ho_bcast = ho_idx[None, :]        # [1, BLOCK_N]
        wo_bcast = wo_idx[None, :]        # [1, BLOCK_N]
        kh_bcast = kh[:, None]            # [BLOCK_K, 1]
        kw_bcast = kw[:, None]            # [BLOCK_K, 1]

        in_h = ho_bcast * stride_h + kh_bcast * dilation_h - pad_h  # [BLOCK_K, BLOCK_N]
        in_w = wo_bcast * stride_w + kw_bcast * dilation_w - pad_w  # [BLOCK_K, BLOCK_N]

        x_ptrs = (
            x_ptr
            + n_bcast * x_stride0
            + ic_bcast * x_stride1
            + in_h * x_stride2
            + in_w * x_stride3
        )

        in_bounds_h = (in_h >= 0) & (in_h < H_in)
        in_bounds_w = (in_w >= 0) & (in_w < W_in)
        mask_x = (
            mask_k[:, None]
            & mask_n[None, :]
            & in_bounds_h
            & in_bounds_w
        )

        x_vals = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # --------- Matrix multiply accumulate ----------
        # w_vals: [BLOCK_M, BLOCK_K]
        # x_vals: [BLOCK_K, BLOCK_N]
        acc += tl.dot(w_vals, x_vals, allow_tf32=True)

    # --------- Add bias if present ----------
    if HAS_BIAS:
        b_ptrs = b_ptr + oc_idx
        b_vals = tl.load(b_ptrs, mask=mask_m, other=0.0)
        acc += b_vals[:, None]

    # --------- Store output Y ----------
    oc_bcast_y = oc_idx[:, None]   # [BLOCK_M, 1]
    n_bcast_y = n_idx[None, :]     # [1, BLOCK_N]
    ho_bcast_y = ho_idx[None, :]   # [1, BLOCK_N]
    wo_bcast_y = wo_idx[None, :]   # [1, BLOCK_N]

    y_ptrs = (
        y_ptr
        + n_bcast_y * y_stride0
        + oc_bcast_y * y_stride1
        + ho_bcast_y * y_stride2
        + wo_bcast_y * y_stride3
    )

    mask_y = (mask_m[:, None]) & (mask_n[None, :])
    tl.store(y_ptrs, acc, mask=mask_y)


def triton_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """
    High-performance 2D convolution using Triton, equivalent to nn.Conv2d forward.
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"

    N, C_in, H_in, W_in = x.shape
    C_out, C_per_group, KH, KW = weight.shape
    assert C_in % groups == 0
    cin_per_group = C_in // groups
    cout_per_group = C_out // groups
    assert C_per_group == cin_per_group

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    # Output spatial sizes (standard conv2d formula)
    H_out = (H_in + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h + 1
    W_out = (W_in + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=torch.float32)

    # Ensure contiguous for predictable strides
    x_c = x.contiguous()
    w_c = weight.contiguous()
    y_c = y  # already contiguous

    N_HW = N * H_out * W_out

    BLOCK_M = 64  # out_channels tile
    BLOCK_N = 64  # output-position tile
    BLOCK_K = 32  # reduction tile (cin_per_group * KH * KW)

    OC_BLOCKS_PER_GROUP = triton.cdiv(cout_per_group, BLOCK_M)

    def grid(meta):
        return (
            groups * OC_BLOCKS_PER_GROUP,
            triton.cdiv(N_HW, meta["BLOCK_N"]),
        )

    has_bias = bias is not None
    if has_bias:
        b_c = bias.contiguous()
    else:
        # Dummy tensor (won't be accessed when HAS_BIAS=False)
        b_c = weight.new_empty(0)

    conv2d_fwd_kernel[grid](
        x_c,
        w_c,
        b_c,
        y_c,
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        KH,
        KW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        cin_per_group,
        cout_per_group,
        N_HW,
        x_c.stride(0),
        x_c.stride(1),
        x_c.stride(2),
        x_c.stride(3),
        w_c.stride(0),
        w_c.stride(1),
        w_c.stride(2),
        w_c.stride(3),
        y_c.stride(0),
        y_c.stride(1),
        y_c.stride(2),
        y_c.stride(3),
        OC_BLOCKS_PER_GROUP,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        HAS_BIAS=has_bias,
        num_warps=4,
        num_stages=3,
    )

    # Cast back to original dtype if needed
    if y_c.dtype != x.dtype:
        y_c = y_c.to(x.dtype)
    return y_c


class ModelNew(nn.Module):
    """
    Triton-accelerated 2D convolution replacement for nn.Conv2d.
    Matches the interface of the original Model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.groups = groups

        # Normalize stride/padding/dilation to 2D tuples
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Parameters mimicking nn.Conv2d layout: (out_channels, in_channels/groups, KH, KW)
        assert in_channels % groups == 0
        cin_per_group = in_channels // groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, cin_per_group, kh, kw)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        # Initialize similarly to default Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = cin_per_group * kh * kw
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


import math  # needed for initialization
