import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W,
    OC, OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """
    NCHW conv2d with stride=1, dilation=1, groups=1.
    Grid:
      pid_hw: tiles over OH*OW
      pid_oc: tiles over OC
      pid_n:  tiles over N

    All output-related fused ops (conv accumulation, bias add, store) share:
      - same grid
      - same (offs_hw, offs_oc) tile
      - same boundary mask 'mask_out'
    """
    pid_hw = tl.program_id(0)  # over OH*OW
    pid_oc = tl.program_id(1)  # over OC
    pid_n = tl.program_id(2)   # over N

    # Tile indices within the output
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    # Base masks for this tile
    mask_hw = offs_hw < (OH * OW)
    mask_oc = offs_oc < OC

    # Map flattened spatial index to (oh, ow)
    oh = offs_hw // OW
    ow = offs_hw % OW

    # Accumulator for output tile: [BLOCK_HW, BLOCK_OC]
    acc = tl.zeros((BLOCK_HW, BLOCK_OC), dtype=tl.float32)

    # Base pointers for this batch
    x_batch_ptr = x_ptr + pid_n * stride_xn

    # Reduction over input channels
    c0 = 0
    while c0 < C_in:
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C_in

        # Weight tile base ptr for this (oc, c) block, KH/KW offset added inside
        w_oc_c_base = (
            w_ptr
            + offs_oc[None, :] * stride_wo
            + offs_c[:, None] * stride_wc
        )

        # Loop over kernel spatial dims – unrolled at compile time
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh[:, None] + kh - PAD_H
                iw = ow[:, None] + kw - PAD_W

                in_bounds_h = (ih >= 0) & (ih < H)
                in_bounds_w = (iw >= 0) & (iw < W)

                # Input mask for this (hw, c) tile
                mask_hw_c = (
                    mask_hw[:, None]
                    & mask_c[None, :]
                    & in_bounds_h
                    & in_bounds_w
                )

                # Input pointers: [BLOCK_HW, BLOCK_C]
                x_ptrs = (
                    x_batch_ptr
                    + ih * stride_xh
                    + iw * stride_xw
                    + offs_c[None, :] * stride_xc
                )
                x_vals = tl.load(x_ptrs, mask=mask_hw_c, other=0.0)

                # Weight pointers: [BLOCK_C, BLOCK_OC]
                w_ptrs = (
                    w_oc_c_base
                    + kh * stride_wkh
                    + kw * stride_wkw
                )
                mask_w = mask_c[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptrs, mask=mask_w, other=0.0)

                # Matmul-style accumulation along C_in
                acc += tl.dot(x_vals, w_vals, allow_tf32=True)

        c0 += BLOCK_C

    # Fused bias add: same output tile (offs_hw, offs_oc), broadcast over HW
    if HAS_BIAS:
        # 1D bias tile per OC, broadcast to [BLOCK_HW, BLOCK_OC]
        bias_vals = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
        acc += bias_vals[None, :]

    # Store output tile – same offsets & mask for all elements in this tile
    y_batch_ptr = y_ptr + pid_n * stride_yn
    y_ptrs = (
        y_batch_ptr
        + offs_oc[None, :] * stride_yc
        + oh[:, None] * stride_yh
        + ow[:, None] * stride_yw
    )
    mask_out = mask_hw[:, None] & mask_oc[None, :]
    tl.store(y_ptrs, acc, mask=mask_out)


def conv2d_triton(x, weight, bias, padding):
    """
    x:      (N, C_in, H, W)
    weight: (OC, C_in, KH, KW)
    bias:   (OC,) or None
    padding: (pad_h, pad_w)
    stride=1, dilation=1, groups=1
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be on CUDA"
    N, C_in, H, W = x.shape
    OC, Cw, KH, KW = weight.shape
    assert Cw == C_in, "Only groups=1 convolutions supported"
    pad_h, pad_w = padding

    OH = H + 2 * pad_h - KH + 1
    OW = W + 2 * pad_w - KW + 1

    y = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wo, stride_wc, stride_wkh, stride_wkw = weight.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    # Tuned for RTX 4090 (SM89): wide tile on HW, medium on OC, modest K tile.
    BLOCK_HW = 128
    BLOCK_OC = 64
    BLOCK_C = 32

    def grid(meta):
        return (
            triton.cdiv(OH * OW, meta["BLOCK_HW"]),
            triton.cdiv(OC, meta["BLOCK_OC"]),
            N,
        )

    has_bias = bias is not None
    b_ptr = bias if has_bias else y  # dummy pointer when no bias

    conv2d_nchw_kernel[grid](
        x, weight, b_ptr, y,
        N, C_in, H, W,
        OC, OH, OW,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wo, stride_wc, stride_wkh, stride_wkw,
        stride_yn, stride_yc, stride_yh, stride_yw,
        KH=KH, KW=KW,
        PAD_H=pad_h, PAD_W=pad_w,
        HAS_BIAS=has_bias,
        BLOCK_HW=BLOCK_HW, BLOCK_OC=BLOCK_OC, BLOCK_C=BLOCK_C,
        num_warps=4,      # good balance for 128x64 tile on Ada
        num_stages=3,     # overlap loads & compute
    )

    return y


@triton.jit
def maxpool2d_3x3s1p1_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_HW: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """
    3x3 maxpool, stride=1, padding=1, NCHW.
    Grid:
      pid_hw: tiles over H*W
      pid_c:  tiles over C
      pid_n:  tiles over N

    All operations (loads from input window, max-reduction, store) share:
      - same grid
      - same (offs_hw, offs_c) tile
      - same boundary mask 'mask_out' for output
    """
    pid_hw = tl.program_id(0)  # over H*W
    pid_c = tl.program_id(1)   # over C
    pid_n = tl.program_id(2)   # over N

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)      # [BLOCK_C]

    mask_hw = offs_hw < (H * W)
    mask_c = offs_c < C

    oh = offs_hw // W
    ow = offs_hw % W

    # Max accumulator for output tile [HW, C]
    max_vals = tl.full((BLOCK_HW, BLOCK_C), -float("inf"), dtype=tl.float32)

    x_batch_ptr = x_ptr + pid_n * stride_xn

    # 3x3 window offsets – unrolled at compile time
    for dh in tl.static_range(-1, 2):
        for dw in tl.static_range(-1, 2):
            ih = oh[:, None] + dh
            iw = ow[:, None] + dw

            in_bounds_h = (ih >= 0) & (ih < H)
            in_bounds_w = (iw >= 0) & (iw < W)

            mask = (
                mask_hw[:, None]
                & mask_c[None, :]
                & in_bounds_h
                & in_bounds_w
            )

            x_ptrs = (
                x_batch_ptr
                + offs_c[None, :] * stride_xc
                + ih * stride_xh
                + iw * stride_xw
            )
            x_vals = tl.load(x_ptrs, mask=mask, other=-float("inf"))
            max_vals = tl.maximum(max_vals, x_vals)

    y_batch_ptr = y_ptr + pid_n * stride_yn
    y_ptrs = (
        y_batch_ptr
        + offs_c[None, :] * stride_yc
        + oh[:, None] * stride_yh
        + ow[:, None] * stride_yw
    )
    mask_out = mask_hw[:, None] & mask_c[None, :]
    tl.store(y_ptrs, max_vals, mask=mask_out)


def maxpool2d_3x3s1p1_triton(x):
    """
    x: (N, C, H, W) -> same shape, kernel=3, stride=1, padding=1 (NCHW)
    """
    assert x.is_cuda, "Input must be on CUDA"
    N, C, H, W = x.shape
    y = torch.empty_like(x)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    BLOCK_HW = 128
    BLOCK_C = 64

    def grid(meta):
        return (
            triton.cdiv(H * W, meta["BLOCK_HW"]),
            triton.cdiv(C, meta["BLOCK_C"]),
            N,
        )

    maxpool2d_3x3s1p1_kernel[grid](
        x, y,
        N, C, H, W,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_yn, stride_yc, stride_yh, stride_yw,
        BLOCK_HW=BLOCK_HW, BLOCK_C=BLOCK_C,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3,
                 reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()

        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1, bias=True)

        # 3x3 convolution branch: 1x1 reduce -> 3x3
        self.branch3x3_reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1, bias=True)
        self.branch3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1, bias=True)

        # 5x5 convolution branch: 1x1 reduce -> 5x5
        self.branch5x5_reduce = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1, bias=True)
        self.branch5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2, bias=True)

        # Max pooling branch: 3x3 maxpool -> 1x1 conv
        self.branch_pool_proj = nn.Conv2d(in_channels, pool_proj, kernel_size=1, bias=True)

    def forward(self, x):
        # 1x1 branch
        branch1x1 = conv2d_triton(
            x,
            self.branch1x1.weight,
            self.branch1x1.bias,
            padding=(0, 0),
        )

        # 3x3 branch
        r3 = conv2d_triton(
            x,
            self.branch3x3_reduce.weight,
            self.branch3x3_reduce.bias,
            padding=(0, 0),
        )
        branch3x3 = conv2d_triton(
            r3,
            self.branch3x3.weight,
            self.branch3x3.bias,
            padding=(1, 1),
        )

        # 5x5 branch
        r5 = conv2d_triton(
            x,
            self.branch5x5_reduce.weight,
            self.branch5x5_reduce.bias,
            padding=(0, 0),
        )
        branch5x5 = conv2d_triton(
            r5,
            self.branch5x5.weight,
            self.branch5x5.bias,
            padding=(2, 2),
        )

        # Pooling branch
        pooled = maxpool2d_3x3s1p1_triton(x)
        branch_pool = conv2d_triton(
            pooled,
            self.branch_pool_proj.weight,
            self.branch_pool_proj.bias,
            padding=(0, 0),
        )

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, dim=1)
