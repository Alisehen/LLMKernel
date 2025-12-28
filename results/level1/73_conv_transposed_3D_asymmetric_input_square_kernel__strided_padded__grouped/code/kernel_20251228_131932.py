import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv_transpose3d_igemm_kernel(
    x_ptr,  # float32 [N, C_IN, D_IN, H_IN, W_IN]
    w_ptr,  # float32 [C_IN, C_OUT_PER_G, K_D, K_H, K_W]
    b_ptr,  # float32 [C_OUT] or dummy
    y_ptr,  # float32 [N, C_OUT, D_OUT, H_OUT, W_OUT]
    N, D_IN, H_IN, W_IN,
    D_OUT, H_OUT, W_OUT,
    C_IN, C_OUT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    K_D: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    C_IN_PER_G: tl.constexpr,
    C_OUT_PER_G: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # Each program:
    #  - axis 0: tile of M = N * D_OUT * H_OUT * W_OUT (spatial * batch) positions
    #  - axis 1: tile of N = C_OUT_PER_G output channels (per group)
    #  - axis 2: group id

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = tl.program_id(2)

    SPATIAL_OUT = D_OUT * H_OUT * W_OUT
    M_TOTAL = N * SPATIAL_OUT

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M_TOTAL

    # Decode (n, d_out, h_out, w_out) from offs_m
    n_idx = offs_m // SPATIAL_OUT
    s_idx = offs_m % SPATIAL_OUT

    wh_out = H_OUT * W_OUT
    d_out = s_idx // wh_out
    rem = s_idx % wh_out
    h_out = rem // W_OUT
    w_out = rem % W_OUT

    # Output channels (local to group)
    oc_local = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = oc_local < C_OUT_PER_G
    oc_global = oc_local + g * C_OUT_PER_G  # global channel index

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Total K dimension per group
    K_TOTAL = C_IN_PER_G * K_D * K_H * K_W

    # Loop over K dimension tiles (implicit GEMM)
    for k_start in tl.static_range(0, K_TOTAL, BLOCK_K):
        k_ids = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_ids < K_TOTAL

        # Map k_ids -> (ci_local, kd, kh, kw)
        tmp = k_ids
        ci_local = tmp // (K_D * K_H * K_W)
        rem_k = tmp % (K_D * K_H * K_W)
        kd = rem_k // (K_H * K_W)
        rem_k = rem_k % (K_H * K_W)
        kh = rem_k // K_W
        kw = rem_k % K_W

        ic_global = g * C_IN_PER_G + ci_local  # [BLOCK_K]

        # Broadcast output coordinates [M,1] with kernel coords [1,K]
        d_out_b = d_out[:, None]
        h_out_b = h_out[:, None]
        w_out_b = w_out[:, None]

        kd_b = kd[None, :]
        kh_b = kh[None, :]
        kw_b = kw[None, :]

        # Compute corresponding input coordinates for transposed conv
        od_nom = d_out_b + PAD_D - kd_b
        oh_nom = h_out_b + PAD_H - kh_b
        ow_nom = w_out_b + PAD_W - kw_b

        valid_d = (od_nom >= 0) & (od_nom < D_IN * STRIDE_D)
        valid_h = (oh_nom >= 0) & (oh_nom < H_IN * STRIDE_H)
        valid_w = (ow_nom >= 0) & (ow_nom < W_IN * STRIDE_W)

        valid_d = valid_d & ((od_nom % STRIDE_D) == 0)
        valid_h = valid_h & ((oh_nom % STRIDE_H) == 0)
        valid_w = valid_w & ((ow_nom % STRIDE_W) == 0)

        id_in = od_nom // STRIDE_D
        ih_in = oh_nom // STRIDE_H
        iw_in = ow_nom // STRIDE_W

        mask_m_b = mask_m[:, None]
        mask_k_b = mask_k[None, :]

        mask_in = mask_m_b & mask_k_b & valid_d & valid_h & valid_w

        # Compute input offsets: ((n * C_IN + ic) * D_IN + id) * H_IN + ih) * W_IN + iw
        n_b = n_idx[:, None]
        ic_b = ic_global[None, :]

        tmp_in = (n_b * C_IN + ic_b) * D_IN + id_in
        tmp_in = tmp_in * H_IN + ih_in
        tmp_in = tmp_in * W_IN + iw_in

        a = tl.load(x_ptr + tmp_in, mask=mask_in, other=0.0)  # [M, K]

        # Compute weight offsets: (((ic * C_OUT_PER_G + oc_local) * K_D + kd) * K_H + kh) * K_W + kw
        icw_b = ic_global[:, None]
        oc_b = oc_local[None, :]

        kd_w = kd[:, None]
        kh_w = kh[:, None]
        kw_w = kw[:, None]

        tmp_w = (icw_b * C_OUT_PER_G + oc_b) * K_D + kd_w
        tmp_w = tmp_w * K_H + kh_w
        tmp_w = tmp_w * K_W + kw_w

        mask_w = mask_k[:, None] & mask_n[None, :]

        b = tl.load(w_ptr + tmp_w, mask=mask_w, other=0.0)  # [K, N]

        # GEMM: [M,K] x [K,N] -> [M,N]
        acc += tl.dot(a, b, allow_tf32=True)

    # Add bias if present
    if HAS_BIAS:
        b_vals = tl.load(b_ptr + oc_global, mask=mask_n, other=0.0)  # [N]
        acc += b_vals[None, :]

    # Store results
    n_b = n_idx[:, None]
    d_b = d_out[:, None]
    h_b = h_out[:, None]
    w_b = w_out[:, None]
    ocg_b = oc_global[None, :]

    tmp_out = (n_b * C_OUT + ocg_b) * D_OUT + d_b
    tmp_out = tmp_out * H_OUT + h_b
    tmp_out = tmp_out * W_OUT + w_b

    mask_out = mask_m[:, None] & mask_n[None, :]

    tl.store(y_ptr + tmp_out, acc, mask=mask_out)


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: tuple,
    padding: tuple,
    groups: int,
    output_padding: tuple,
) -> torch.Tensor:
    """
    x:       (N, C_in, D_in, H_in, W_in)
    weight:  (C_in, C_out_per_group, K_D, K_H, K_W)
    bias:    (C_out,) or None
    stride:  (s_d, s_h, s_w)
    padding: (p_d, p_h, p_w)
    groups:  int
    output_padding: (op_d, op_h, op_w) -- currently only all zeros supported
    """
    assert x.is_cuda and weight.is_cuda, "Triton conv_transpose3d requires CUDA tensors"

    # For simplicity and correctness, only support output_padding == 0 in this kernel
    op_d, op_h, op_w = output_padding
    assert op_d == 0 and op_h == 0 and op_w == 0, "Non-zero output_padding is not supported in Triton kernel"

    device = x.device
    x_fp32 = x.contiguous().to(torch.float32)
    w_fp32 = weight.contiguous().to(torch.float32)

    has_bias = bias is not None
    b_fp32 = bias.contiguous().to(torch.float32) if has_bias else x_fp32  # dummy if no bias

    N, C_in, D_in, H_in, W_in = x_fp32.shape
    C_in_w, C_out_per_g, K_D, K_H, K_W = w_fp32.shape
    assert C_in_w == C_in, "weight C_in mismatch"

    s_d, s_h, s_w = stride
    p_d, p_h, p_w = padding

    C_out = C_out_per_g * groups

    # Transposed conv output dimensions (PyTorch formula, dilation = 1)
    D_out = (D_in - 1) * s_d - 2 * p_d + K_D + op_d
    H_out = (H_in - 1) * s_h - 2 * p_h + K_H + op_h
    W_out = (W_in - 1) * s_w - 2 * p_w + K_W + op_w

    y_fp32 = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=device,
        dtype=torch.float32,
    )

    C_in_per_g = C_in // groups
    C_out_per_g = C_out // groups

    SPATIAL_OUT = D_out * H_out * W_out
    M_TOTAL = N * SPATIAL_OUT

    # Tuning parameters
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 32

    grid_m = triton.cdiv(M_TOTAL, BLOCK_M)
    grid_n = triton.cdiv(C_out_per_g, BLOCK_N)
    grid = (grid_m, grid_n, groups)

    conv_transpose3d_igemm_kernel[grid](
        x_fp32,
        w_fp32,
        b_fp32,
        y_fp32,
        N,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        C_in,
        C_out,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        STRIDE_D=s_d,
        STRIDE_H=s_h,
        STRIDE_W=s_w,
        PAD_D=p_d,
        PAD_H=p_h,
        PAD_W=p_w,
        K_D=K_D,
        K_H=K_H,
        K_W=K_W,
        C_IN_PER_G=C_in_per_g,
        C_OUT_PER_G=C_out_per_g,
        GROUPS=groups,
        HAS_BIAS=has_bias,
    )

    if x.dtype != torch.float32:
        return y_fp32.to(x.dtype)
    return y_fp32


class ModelNew(nn.Module):
    """
    ConvTranspose3d implemented with a high-performance Triton kernel.
    Matches the API and initialization of the original PyTorch module
    for the case output_padding == 0.
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
        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias
        stride = self.conv_transpose3d.stride
        padding = self.conv_transpose3d.padding
        groups = self.conv_transpose3d.groups
        output_padding = self.conv_transpose3d.output_padding

        return triton_conv_transpose3d(x, w, b, stride, padding, groups, output_padding)
