import torch
import torch.nn as nn
import triton
import triton.language as tl


# =========================
# Optimized Triton Conv2D (NCHW, stride=1, generic KxK, padding)
# Implemented as GEMM with on-the-fly im2col
# =========================
@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "C_out", "K"],
)
@triton.jit
def conv2d_nchw_kernel(
    x_ptr, w_flat_ptr, b_ptr, y_ptr,
    N, C_in, H_in, W_in,
    H_out, W_out,
    C_out,
    KH, KW,
    pad_h, pad_w,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    stride_wk, stride_wo,
    M, K,  # M = N * H_out * W_out, K = C_in * KH * KW
    BLOCK_M: tl.constexpr,  # tile over M (N*H_out*W_out)
    BLOCK_N: tl.constexpr,  # tile over C_out
    BLOCK_K: tl.constexpr,  # reduction tile over K
):
    # Program IDs for 2D tiling of output matrix [M, C_out]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C_out

    # Map linear M index -> (n_idx, h_out_idx, w_out_idx)
    HW_out = H_out * W_out

    n_idx = offs_m // HW_out
    rem_m = offs_m - n_idx * HW_out
    h_out_idx = rem_m // W_out
    w_out_idx = rem_m - h_out_idx * W_out

    # Base spatial indices after padding adjustment (broadcasted later)
    h_base = h_out_idx - pad_h
    w_base = w_out_idx - pad_w

    # FP32 accumulator tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pre-broadcasted output indices
    n_b = n_idx[:, None]
    h0_b = h_base[:, None]
    w0_b = w_base[:, None]

    # Reduction loop over K = C_in * KH * KW
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        tl.multiple_of(offs_k, BLOCK_K)
        k_mask = offs_k < K

        # Map linear K index -> (c_idx, kh_idx, kw_idx)
        tmp = offs_k // C_in
        c_idx = offs_k - tmp * C_in
        kh_idx = tmp // KW
        kw_idx = tmp - kh_idx * KW

        c_b = c_idx[None, :]
        kh_b = kh_idx[None, :]
        kw_b = kw_idx[None, :]

        h_in = h0_b + kh_b
        w_in = w0_b + kw_b

        # In-bounds mask for input
        in_bounds = (
            m_mask[:, None] & k_mask[None, :] &
            (h_in >= 0) & (h_in < H_in) &
            (w_in >= 0) & (w_in < W_in)
        )

        # Input pointers [BLOCK_M, BLOCK_K]
        x_ptrs = (
            x_ptr
            + n_b * stride_x_n
            + c_b * stride_x_c
            + h_in * stride_x_h
            + w_in * stride_x_w
        )

        a = tl.load(x_ptrs, mask=in_bounds, other=0.0)

        # Weights pointers [BLOCK_K, BLOCK_N], w_flat is [K, C_out]
        w_ptrs = (
            w_flat_ptr
            + offs_k[:, None] * stride_wk
            + offs_n[None, :] * stride_wo
        )
        w_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Matmul accumulate in FP32 (Tensor Cores via allow_tf32)
        acc += tl.dot(a, b, allow_tf32=True)

    # Bias add (fused on same (offs_m, offs_n) grid)
    bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    bias = bias.to(tl.float32)
    acc += bias[None, :]

    # Store back to y (NCHW layout)
    h_out_b = h_out_idx[:, None]
    w_out_b = w_out_idx[:, None]
    oc_b = offs_n[None, :]

    y_ptrs = (
        y_ptr
        + n_b * stride_y_n
        + oc_b * stride_y_c
        + h_out_b * stride_y_h
        + w_out_b * stride_y_w
    )

    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


def conv2d_triton(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, padding: int):
    """
    High-performance NCHW Conv2d with stride=1, arbitrary kernel, and symmetric padding.
    x: (N, C_in, H_in, W_in)
    weight: (C_out, C_in, KH, KW)
    bias: (C_out,)
    padding: int or (pad_h, pad_w)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in_w == C_in, "C_in mismatch"

    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    stride_h = stride_w = 1
    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Flatten weights to [K, C_out]
    K = C_in * KH * KW
    M = N * H_out * W_out
    w_flat = weight.view(C_out, K).transpose(0, 1).contiguous()

    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_y_n, stride_y_c, stride_y_h, stride_y_w = y.stride()
    stride_wk, stride_wo = w_flat.stride()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_nchw_kernel[grid](
        x, w_flat, bias, y,
        N, C_in, H_in, W_in,
        H_out, W_out,
        C_out,
        KH, KW,
        pad_h, pad_w,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
        stride_wk, stride_wo,
        M, K,
    )

    return y


# =========================
# Optimized Triton MaxPool2d 3x3, stride=1, padding=1 (NCHW)
# 1D grid over all output elements
# =========================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK': 512}, num_warps=4, num_stages=2),
    ],
    key=["NUMEL"],
)
@triton.jit
def maxpool2d_3x3_s1_p1_nchw_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    NUMEL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NUMEL

    # Map linear index -> (n, c, h, w)
    CHW = C * H * W
    HW = H * W

    n = offs // CHW
    rem1 = offs - n * CHW
    c = rem1 // HW
    rem2 = rem1 - c * HW
    h = rem2 // W
    w = rem2 - h * W

    neg_inf = -float("inf")
    max_val = tl.full((BLOCK,), neg_inf, dtype=tl.float32)

    # 3x3 window (padding=1, so we just bounds-check)
    for dh in range(-1, 2):
        for dw in range(-1, 2):
            h_in = h + dh
            w_in = w + dw

            in_bounds = (
                mask &
                (h_in >= 0) & (h_in < H) &
                (w_in >= 0) & (w_in < W)
            )

            x_ptrs = (
                x_ptr
                + n * stride_x_n
                + c * stride_x_c
                + h_in * stride_x_h
                + w_in * stride_x_w
            )

            vals = tl.load(x_ptrs, mask=in_bounds, other=neg_inf)
            vals = vals.to(tl.float32)
            max_val = tl.maximum(max_val, vals)

    y_ptrs = (
        y_ptr
        + n * stride_y_n
        + c * stride_y_c
        + h * stride_y_h
        + w * stride_y_w
    )

    tl.store(y_ptrs, max_val, mask=mask)


def maxpool2d_3x3_s1_p1_triton(x: torch.Tensor):
    """
    High-performance MaxPool2d with kernel_size=3, stride=1, padding=1 for NCHW.
    x: (N, C, H, W)
    """
    assert x.is_cuda
    x = x.contiguous()
    N, C, H, W = x.shape
    y = torch.empty_like(x)

    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_y_n, stride_y_c, stride_y_h, stride_y_w = y.stride()

    NUMEL = N * C * H * W

    grid = lambda META: (
        triton.cdiv(NUMEL, META["BLOCK"]),
    )

    maxpool2d_3x3_s1_p1_nchw_kernel[grid](
        x, y,
        N, C, H, W,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
        NUMEL,
    )

    return y


# =========================
# Full Model using optimized Triton kernels
# =========================
class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_1x1,
        reduce_3x3,
        out_3x3,
        reduce_5x5,
        out_5x5,
        pool_proj,
    ):
        super(ModelNew, self).__init__()

        # Keep the same module structure / parameter layout as the original
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        # 1x1 convolution branch
        w1 = self.branch1x1.weight
        b1 = self.branch1x1.bias
        branch1x1 = conv2d_triton(x, w1, b1, padding=0)

        # 3x3 convolution branch: 1x1 reduce -> 3x3 conv
        w3r = self.branch3x3[0].weight
        b3r = self.branch3x3[0].bias
        reduce3x3 = conv2d_triton(x, w3r, b3r, padding=0)

        w3 = self.branch3x3[1].weight
        b3 = self.branch3x3[1].bias
        branch3x3 = conv2d_triton(reduce3x3, w3, b3, padding=1)

        # 5x5 convolution branch: 1x1 reduce -> 5x5 conv
        w5r = self.branch5x5[0].weight
        b5r = self.branch5x5[0].bias
        reduce5x5 = conv2d_triton(x, w5r, b5r, padding=0)

        w5 = self.branch5x5[1].weight
        b5 = self.branch5x5[1].bias
        branch5x5 = conv2d_triton(reduce5x5, w5, b5, padding=2)

        # Max pooling branch: 3x3 maxpool -> 1x1 conv
        pool = maxpool2d_3x3_s1_p1_triton(x)
        wp = self.branch_pool[1].weight
        bp = self.branch_pool[1].bias
        branch_pool = conv2d_triton(pool, wp, bp, padding=0)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, dim=1)
