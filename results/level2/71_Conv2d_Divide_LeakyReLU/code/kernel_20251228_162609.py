import torch, torch.nn as nn, triton, triton.language as tl
import math


@triton.jit
def conv2d_div_leakyrelu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, H_in, W_in,
    H_out, W_out,
    Cin, Cout,
    KH, KW,
    M, N, K,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    divisor, negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling over (M, N) = (B * H_out * W_out, Cout)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Map flattened M index -> (b, ho, wo)
    HW_out = H_out * W_out
    b_idx = offs_m // HW_out
    rem_m = offs_m % HW_out
    ho_idx = rem_m // W_out
    wo_idx = rem_m % W_out

    # Accumulator in FP32 for better precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K dimension is flattened over (Cin, KH, KW)
    k_range = tl.arange(0, BLOCK_K)
    KHW = KH * KW

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + k_range  # [BLOCK_K]
        k_mask = offs_k < K

        # Map flattened K index -> (ci, kh, kw)
        ci = offs_k // KHW
        rem_k = offs_k % KHW
        kh = rem_k // KW
        kw = rem_k % KW

        # Build input pointers for A: shape (BLOCK_M, BLOCK_K)
        # x[b, ci, ho + kh, wo + kw]
        b_b = b_idx[:, None]
        ho_b = ho_idx[:, None]
        wo_b = wo_idx[:, None]

        ci_b = ci[None, :]
        kh_b = kh[None, :]
        kw_b = kw[None, :]

        h_in = ho_b + kh_b
        w_in = wo_b + kw_b

        a_ptrs = (
            x_ptr
            + b_b * stride_x_n
            + ci_b * stride_x_c
            + h_in * stride_x_h
            + w_in * stride_x_w
        )

        a_mask = (mask_m[:, None]) & (k_mask[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Build weight pointers for B: shape (BLOCK_K, BLOCK_N)
        # w[co, ci, kh, kw]
        co_b = offs_n[None, :]
        ci_k = ci[:, None]
        kh_k = kh[:, None]
        kw_k = kw[:, None]

        w_ptrs = (
            w_ptr
            + co_b * stride_w_cout
            + ci_k * stride_w_cin
            + kh_k * stride_w_kh
            + kw_k * stride_w_kw
        )

        b_mask = (k_mask[:, None]) & (mask_n[None, :])
        w = tl.load(w_ptrs, mask=b_mask, other=0.0)

        # Matrix multiply-accumulate
        acc += tl.dot(a, w, allow_tf32=True)

    # Add bias per output channel
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias_vals[None, :]

    # Divide by constant
    acc = acc / divisor

    # LeakyReLU: x if x >= 0 else negative_slope * x
    acc = tl.where(acc >= 0, acc, acc * negative_slope)

    # Store results
    # Map (offs_m, offs_n) -> (b, co, ho, wo)
    out_b = b_idx[:, None]
    out_co = offs_n[None, :]
    out_ho = ho_idx[:, None]
    out_wo = wo_idx[:, None]

    y_ptrs = (
        y_ptr
        + out_b * stride_y_n
        + out_co * stride_y_c
        + out_ho * stride_y_h
        + out_wo * stride_y_w
    )

    out_mask = (mask_m[:, None]) & (mask_n[None, :])
    tl.store(y_ptrs, acc, mask=out_mask)


def fused_conv2d_div_leakyrelu(x, weight, bias, divisor, negative_slope=0.01):
    """
    Fused implementation of:
        y = conv2d(x, weight, bias, stride=1, padding=0)
        y = y / divisor
        y = leaky_relu(y, negative_slope)
    Assumes NCHW layout and groups=1, dilation=1, stride=1, padding=0.
    """
    assert x.ndim == 4, "Input must be 4D NCHW"
    B, Cin, H_in, W_in = x.shape
    Cout, Cin_w, KH, KW = weight.shape
    assert Cin == Cin_w, "Incompatible input/weight channels"
    assert bias is not None and bias.numel() == Cout

    # Output dimensions for stride=1, padding=0
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1
    assert H_out > 0 and W_out > 0, "Invalid kernel size for given input"

    # Flattened GEMM dimensions
    M = B * H_out * W_out
    N = Cout
    K = Cin * KH * KW

    y = torch.empty((B, Cout, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides (in elements)
    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x.stride()
    stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw = weight.stride()
    stride_y_n, stride_y_c, stride_y_h, stride_y_w = y.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    conv2d_div_leakyrelu_kernel[grid](
        x, weight, bias, y,
        B, H_in, W_in,
        H_out, W_out,
        Cin, Cout,
        KH, KW,
        M, N, K,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
        float(divisor), float(negative_slope),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated version of:
        Conv2d -> divide by constant -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.divisor = divisor

        k = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, k, k))
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Initialize like nn.Conv2d default (Kaiming uniform)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * k * k
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_conv2d_div_leakyrelu(
            x, self.weight, self.bias, self.divisor, negative_slope=0.01
        )
