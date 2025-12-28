import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_gemm_kernel(
    x_ptr,          # *float32, [N, C_in, D_in, H_in, W_in]
    wcol_ptr,       # *float32, [K, C_out] where K = C_in * Kd * Kh * Kw
    b_ptr,          # *float32, [C_out]
    out_ptr,        # *float32, [N, C_out, D_out, H_out, W_out]
    N,
    C_in,
    C_out,
    D_in,
    H_in,
    W_in,
    D_out,
    H_out,
    W_out,
    Kd,
    Kh,
    Kw,
    M,             # M = N * D_out * H_out * W_out (number of output voxels per channel)
    K,             # K = C_in * Kd * Kh * Kw
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for tiling over [M, C_out]
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode offs_m -> (n, od, oh, ow)
    tmp = offs_m
    ow = tmp % W_out
    tmp = tmp // W_out
    oh = tmp % H_out
    tmp = tmp // H_out
    od = tmp % D_out
    n = tmp // D_out  # [BLOCK_M]

    # Initialize accumulator with bias
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc += b[None, :]  # Broadcast bias along BLOCK_M

    # Precompute spatial products
    KdKhKw = Kd * Kh * Kw
    KhKw = Kh * Kw

    # GEMM-like loop over K dimension (ic, kd, kh, kw)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K  # [BLOCK_K]

        # Decode offs_k -> (ic, kd, kh, kw)
        ic = offs_k // KdKhKw
        rem = offs_k % KdKhKw
        kd = rem // KhKw
        rem2 = rem % KhKw
        kh = rem2 // Kw
        kw = rem2 % Kw
        # Shapes: [BLOCK_K] each

        # Compute input coordinates for this (m, k) tile:
        # id = od - kd, ih = oh - kh, iw = ow - kw
        od_mat = od[:, None]
        oh_mat = oh[:, None]
        ow_mat = ow[:, None]
        kd_mat = kd[None, :]
        kh_mat = kh[None, :]
        kw_mat = kw[None, :]

        id_ = od_mat - kd_mat
        ih_ = oh_mat - kh_mat
        iw_ = ow_mat - kw_mat

        # Validity mask for input coords
        valid_d = (id_ >= 0) & (id_ < D_in)
        valid_h = (ih_ >= 0) & (ih_ < H_in)
        valid_w = (iw_ >= 0) & (iw_ < W_in)
        valid_mk = (
            valid_d & valid_h & valid_w
            & mask_m[:, None]
            & mask_k[None, :]
        )

        # Compute linear indices into x: [N, C_in, D_in, H_in, W_in]
        n_mat = n[:, None]
        ic_mat = ic[None, :]

        x_idx = (
            (((n_mat * C_in + ic_mat) * D_in + id_) * H_in + ih_) * W_in + iw_
        )

        x_tile = tl.load(x_ptr + x_idx, mask=valid_mk, other=0.0)  # [BM, BK]

        # Load corresponding weights for this K-tile:
        # wcol is [K, C_out] laid out as row-major, so index = k * C_out + oc
        k_mat = offs_k[:, None]
        oc_mat = offs_n[None, :]
        w_idx = k_mat * C_out + oc_mat  # [BK, BN]
        valid_kn = mask_k[:, None] & mask_n[None, :]

        w_tile = tl.load(wcol_ptr + w_idx, mask=valid_kn, other=0.0)  # [BK, BN]

        # GEMM: [BM, BK] x [BK, BN] -> [BM, BN]
        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # Write back to output tensor
    # out: [N, C_out, D_out, H_out, W_out]
    n_mat = n[:, None]
    od_mat = od[:, None]
    oh_mat = oh[:, None]
    ow_mat = ow[:, None]
    oc_mat = offs_n[None, :]

    out_idx = (
        (((n_mat * C_out + oc_mat) * D_out + od_mat) * H_out + oh_mat) * W_out + ow_mat
    )

    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_idx, acc, mask=mask_out)


def conv_transpose3d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride,
    padding,
    output_padding,
    dilation,
    groups: int,
) -> torch.Tensor:
    """
    High-performance ConvTranspose3d using a fused vol2col+GEMM formulation in Triton.

    Supported configuration (matching the original optimized case):
      - groups == 1
      - stride == (1, 1, 1)
      - padding == (0, 0, 0)
      - dilation == (1, 1, 1)
      - output_padding == (0, 0, 0)
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dtype == torch.float32, "This Triton kernel currently supports float32 only"

    # Normalize hyperparameters to 3-tuples
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    # Enforce the supported configuration
    assert groups == 1, "This Triton ConvTranspose3d currently supports groups == 1 only"
    assert stride == (1, 1, 1), "Only stride == 1 is supported in this Triton kernel"
    assert padding == (0, 0, 0), "Only padding == 0 is supported in this Triton kernel"
    assert dilation == (1, 1, 1), "Only dilation == 1 is supported in this Triton kernel"
    assert output_padding == (0, 0, 0), "Only output_padding == 0 is supported in this Triton kernel"

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is None:
        bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)
    else:
        bias = bias.contiguous()

    N, C_in, D_in, H_in, W_in = x.shape
    C_in_w, C_out, Kd, Kh, Kw = weight.shape
    assert C_in_w == C_in, "weight shape mismatch with input channels"

    # Output dimensions for stride=1, padding=0, dilation=1, output_padding=0
    D_out = D_in + (Kd - 1)
    H_out = H_in + (Kh - 1)
    W_out = W_in + (Kw - 1)

    out = torch.empty((N, C_out, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # GEMM dimensions
    M = N * D_out * H_out * W_out
    K = C_in * Kd * Kh * Kw

    # Re-layout weight as [K, C_out] to improve GEMM access pattern
    # Original: [C_in, C_out, Kd, Kh, Kw]
    w_col = (
        weight.view(C_in, C_out, Kd * Kh * Kw)
        .permute(0, 2, 1)                # [C_in, Kspatial, C_out]
        .contiguous()
        .view(K, C_out)                  # [K, C_out]
    )

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(C_out, meta["BLOCK_N"]),
    )

    conv_transpose3d_gemm_kernel[grid](
        x,
        w_col,
        bias,
        out,
        N,
        C_in,
        C_out,
        D_in,
        H_in,
        W_in,
        D_out,
        H_out,
        W_out,
        Kd,
        Kh,
        Kw,
        M,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for the provided ConvTranspose3d-based model.
    Uses a custom ConvTranspose3d implementation in Triton for the forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        # Keep a PyTorch ConvTranspose3d module to hold weights / initialization,
        # but we will NOT use its forward; only its parameters.
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            (kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Cache hyperparameters for the Triton implementation
        self.stride = self.conv_transpose3d.stride
        self.padding = self.conv_transpose3d.padding
        self.output_padding = self.conv_transpose3d.output_padding
        self.dilation = self.conv_transpose3d.dilation
        self.groups = self.conv_transpose3d.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias

        return conv_transpose3d_triton(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )
