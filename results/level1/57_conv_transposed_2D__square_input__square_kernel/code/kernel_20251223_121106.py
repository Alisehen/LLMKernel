import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv2d_nchw_kernel(
    x_ptr,            # float*  [N, C_in, H_in, W_in]
    w_ptr,            # float*  [C_out, C_in, K_h, K_w]  (flipped+permuted)
    b_ptr,            # float*  [C_out]
    y_ptr,            # float*  [N, C_out, H_out, W_out]
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    K_h, K_w,
    PAD_H, PAD_W,
    K_TOTAL,          # C_in * K_h * K_w
    M,                # N * H_out * W_out
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for tiles along output "rows" (M = N*H_out*W_out) and channels (C_out)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Map "row" index m -> (n, h_out, w_out)
    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    h_out = rem // W_out
    w_out = rem % W_out

    # Broadcasted versions for tile computations
    row_n = n_idx[:, None]        # [BM, 1]
    row_ho = h_out[:, None]       # [BM, 1]
    row_wo = w_out[:, None]       # [BM, 1]
    col_co = offs_n[None, :]      # [1, BN]

    # Strides for contiguous NCHW tensors
    HW_in = H_in * W_in
    stride_xn = C_in * HW_in
    stride_xc = HW_in
    stride_xh = W_in
    stride_xw = 1

    HW_out_full = H_out * W_out
    stride_yn = C_out * HW_out_full
    stride_yc = HW_out_full
    stride_yh = W_out
    stride_yw = 1

    stride_wc_out = C_in * K_h * K_w
    stride_wc_in = K_h * K_w
    stride_wkh = K_w
    stride_wkw = 1

    # Initialize accumulator with bias
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Load bias for columns and broadcast across rows
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b[None, :]

    # Reduction over K dimension = C_in * K_h * K_w
    kk_size = K_h * K_w
    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_TOTAL

        # Map k -> (c_in, k_h, k_w)
        ci = offs_k // kk_size
        remk = offs_k % kk_size
        kh = remk // K_w
        kw = remk % K_w

        # Broadcasted kernel indices
        ci_bc = ci[None, :]        # [1, BK]
        kh_bc = kh[None, :]        # [1, BK]
        kw_bc = kw[None, :]        # [1, BK]

        # Input coordinates for this output tile and K-block
        h_in = row_ho - PAD_H + kh_bc   # [BM, BK]
        w_in = row_wo - PAD_W + kw_bc   # [BM, BK]

        # Broadcast batch and channel indices
        n_bc = row_n                    # [BM, 1]

        # Compute flat indices into x
        x_idx = (
            n_bc * stride_xn
            + ci_bc * stride_xc
            + h_in * stride_xh
            + w_in * stride_xw
        )  # [BM, BK]

        # Validity mask for x
        valid_h = (h_in >= 0) & (h_in < H_in)
        valid_w = (w_in >= 0) & (w_in < W_in)
        mask_x = valid_h & valid_w
        mask_x = mask_x & mask_m[:, None] & mask_k[None, :]

        x_tile = tl.load(x_ptr + x_idx, mask=mask_x, other=0.0)

        # Weight tile [BK, BN]
        ci_k = ci[:, None]      # [BK, 1]
        kh_k = kh[:, None]      # [BK, 1]
        kw_k = kw[:, None]      # [BK, 1]

        w_idx = (
            col_co * stride_wc_out
            + ci_k * stride_wc_in
            + kh_k * stride_wkh
            + kw_k * stride_wkw
        )  # [BK, BN]

        mask_w = mask_n[None, :] & mask_k[:, None]
        w_tile = tl.load(w_ptr + w_idx, mask=mask_w, other=0.0)

        # Accumulate dot product along K
        acc += tl.dot(x_tile, w_tile, allow_tf32=True)

    # Store results to y
    y_idx = (
        n_idx[:, None] * stride_yn
        + offs_n[None, :] * stride_yc
        + h_out[:, None] * stride_yh
        + w_out[:, None] * stride_yw
    )  # [BM, BN]

    mask_y = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptr + y_idx, acc, mask=mask_y)


def triton_conv_transpose2d_nchw_stride1_pad0(
    x: torch.Tensor,
    weight: torch.Tensor,  # [C_in, C_out, K_h, K_w]
    bias: torch.Tensor or None,
) -> torch.Tensor:
    """
    High-performance ConvTranspose2d for:
        - NCHW layout
        - stride = (1, 1)
        - padding = (0, 0)
        - output_padding = (0, 0)
        - dilation = (1, 1)
        - groups = 1
    Implemented via equivalent Conv2d with flipped/permuted weights.
    """
    assert x.is_cuda, "Triton kernel requires CUDA tensor"
    x = x.contiguous()
    weight = weight.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out, K_h, K_w = weight.shape
    assert C_in_w == C_in, "Input channels mismatch between x and weight"

    if bias is None:
        bias = torch.zeros(C_out, device=x.device, dtype=x.dtype)
    else:
        bias = bias.contiguous()

    # Equivalent Conv2d params:
    #   y = conv2d(x, V, padding=(K_h-1, K_w-1))
    # where V[co, ci, kh, kw] = weight[ci, co, K_h-1-kh, K_w-1-kw]
    V = (
        weight.permute(1, 0, 2, 3)  # [C_out, C_in, K_h, K_w]
        .flip(-1, -2)               # flip spatial dims
        .contiguous()
    )

    PAD_H = K_h - 1
    PAD_W = K_w - 1

    H_out = H_in + K_h - 1
    W_out = W_in + K_w - 1

    y = torch.empty(
        (N, C_out, H_out, W_out),
        device=x.device,
        dtype=x.dtype,
    )

    K_total = C_in * K_h * K_w
    M = N * H_out * W_out

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(C_out, meta["BLOCK_N"]),
    )

    conv2d_nchw_kernel[grid](
        x,
        V,
        bias,
        y,
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        H_out,
        W_out,
        K_h,
        K_w,
        PAD_H,
        PAD_W,
        K_total,
        M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated version of the given ConvTranspose2d model.
    Uses a high-performance custom kernel for the common case:
      - stride = 1
      - padding = 0
      - output_padding = 0
      - dilation = 1
      - groups = 1
      - float32, NCHW, CUDA tensors
    Falls back to PyTorch's ConvTranspose2d otherwise.
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
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv_transpose2d

        # Conditions for Triton fast path
        use_triton = (
            x.is_cuda
            and x.dtype == torch.float32
            and conv.stride == (1, 1)
            and conv.padding == (0, 0)
            and conv.output_padding == (0, 0)
            and conv.dilation == (1, 1)
            and conv.groups == 1
        )

        if not use_triton:
            return conv(x)

        return triton_conv_transpose2d_nchw_stride1_pad0(
            x,
            conv.weight,
            conv.bias,
        )
