import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline: balanced, low register pressure, good for multi-input fusion
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        # More rows (M) per program: better when M is large and N is moderate
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        # More columns (N) per program: better when Cout is large
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=2,
        ),
        # High-parallelism option when register pressure is low
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def conv3d_fused_tanh_sigmoid_kernel(
    x_ptr,                # *f32, input  (B, Cin, Din, Hin, Win)
    w_ptr,                # *f32, weight (K, Cout)  where K = Cin*Kd*Kh*Kw
    conv_bias_ptr,        # *f32, (Cout,)
    scale_ptr,            # *f32, (Cout,)
    bias_ptr,             # *f32, (Cout,)
    out_ptr,              # *f32, output (B, Cout, Dout, Hout, Wout)
    B, Cin,
    Din, Hin, Win,
    Cout,
    Kd, Kh, Kw,
    Dout, Hout, Wout,
    stride_xb, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wk, stride_wn,
    stride_ob, stride_oc, stride_od, stride_oh, stride_ow,
    M, N, K,              # M = B*Dout*Hout*Wout, N = Cout, K = Cin*Kd*Kh*Kw
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # 2D tiling over output matrix [M, N]
    # -------------------------------------------------------------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    mask_m = offs_m < M
    mask_n = offs_n < N

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # -------------------------------------------------------------------------
    # Decompose M index -> (b, dout, hout, wout)
    # done once per tile, reused for both A-loads and out-store
    # -------------------------------------------------------------------------
    tmp_m = offs_m
    wout = (tmp_m % Wout)[:, None]       # [BM, 1]
    tmp_m = tmp_m // Wout
    hout = (tmp_m % Hout)[:, None]       # [BM, 1]
    tmp_m = tmp_m // Hout
    dout = (tmp_m % Dout)[:, None]       # [BM, 1]
    tmp_m = tmp_m // Dout
    b_idx = tmp_m[:, None]               # [BM, 1]

    # Base input pointer for this (b, dout, hout, wout) tile
    base_x_ptrs = (
        x_ptr
        + b_idx * stride_xb
        + dout * stride_xd
        + hout * stride_xh
        + wout * stride_xw
    )  # [BM, 1]

    # -------------------------------------------------------------------------
    # Accumulator
    # -------------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # -------------------------------------------------------------------------
    # K-loop: GEMM-style reduction over K = Cin * Kd * Kh * Kw
    # -------------------------------------------------------------------------
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k  # [BK]
        mask_k = k_idx < K

        # Decompose K index -> (cin, kd, kh, kw)
        tmp_k = k_idx
        kw_idx = (tmp_k % Kw)[None, :]       # [1, BK]
        tmp_k = tmp_k // Kw
        kh_idx = (tmp_k % Kh)[None, :]       # [1, BK]
        tmp_k = tmp_k // Kh
        kd_idx = (tmp_k % Kd)[None, :]       # [1, BK]
        tmp_k = tmp_k // Kd
        cin_idx = tmp_k[None, :]             # [1, BK]

        # Input tile A (x)
        x_ptrs = (
            base_x_ptrs
            + cin_idx * stride_xc
            + kd_idx * stride_xd
            + kh_idx * stride_xh
            + kw_idx * stride_xw
        )  # [BM, BK]

        mask_a = mask_m[:, None] & mask_k[None, :]
        a = tl.load(x_ptrs, mask=mask_a, other=0.0)

        # Weight tile B (w)
        w_ptrs = (
            w_ptr
            + k_idx[:, None] * stride_wk
            + offs_n[None, :] * stride_wn
        )  # [BK, BN]

        mask_b = mask_k[:, None] & mask_n[None, :]
        b = tl.load(w_ptrs, mask=mask_b, other=0.0)

        # FMA on Tensor Cores (TF32) where possible
        acc += tl.dot(a, b, allow_tf32=True)

    # -------------------------------------------------------------------------
    # Fused epilogue: conv_bias -> scale -> tanh -> bias -> sigmoid
    # Single output store; all intermediates stay in registers.
    # -------------------------------------------------------------------------
    conv_bias = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)  # [BN]
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0)          # [BN]
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)            # [BN]

    # conv bias and per-channel scale
    acc = acc + conv_bias[None, :]
    acc = acc * scale[None, :]

    # tanh(x) = (e^{2x} - 1) / (e^{2x} + 1)
    t = tl.exp(2.0 * acc)
    acc = (t - 1.0) / (t + 1.0)

    # multiply by per-channel bias
    acc = acc * bias[None, :]

    # sigmoid(x) = 1 / (1 + e^{-x})
    t = tl.exp(-acc)
    acc = 1.0 / (1.0 + t)

    # -------------------------------------------------------------------------
    # Store final output
    # -------------------------------------------------------------------------
    cout = offs_n[None, :]  # [1, BN]

    out_ptrs = (
        out_ptr
        + b_idx * stride_ob
        + cout * stride_oc
        + dout * stride_od
        + hout * stride_oh
        + wout * stride_ow
    )  # [BM, BN]

    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


def fused_conv3d_tanh_sigmoid(x, weight, conv_bias, scaling_factor, bias_param):
    """
    x:              (B, Cin, Din, Hin, Win)
    weight:         (Cout, Cin, Kd, Kh, Kw)
    conv_bias:      (Cout,)
    scaling_factor: (Cout, 1, 1, 1)
    bias_param:     (Cout, 1, 1, 1)
    """
    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()
    scaling_factor = scaling_factor.contiguous().view(-1)
    bias_param = bias_param.contiguous().view(-1)

    B, Cin, Din, Hin, Win = x.shape
    Cout, Cin_w, Kd, Kh, Kw = weight.shape
    assert Cin == Cin_w, "Input channels must match weight channels"

    # Valid 3D conv, stride=1, padding=0
    Dout = Din - Kd + 1
    Hout = Hin - Kh + 1
    Wout = Win - Kw + 1

    # Flatten weight to (K, Cout)
    K = Cin * Kd * Kh * Kw
    weight_flat = weight.view(Cout, K).transpose(0, 1).contiguous()

    # Output allocation
    out = torch.empty((B, Cout, Dout, Hout, Wout), device=x.device, dtype=x.dtype)

    stride_xb, stride_xc, stride_xd, stride_xh, stride_xw = x.stride()
    stride_ob, stride_oc, stride_od, stride_oh, stride_ow = out.stride()
    stride_wk, stride_wn = weight_flat.stride()

    M = B * Dout * Hout * Wout
    N = Cout

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    conv3d_fused_tanh_sigmoid_kernel[grid](
        x,
        weight_flat,
        conv_bias,
        scaling_factor,
        bias_param,
        out,
        B,
        Cin,
        Din,
        Hin,
        Win,
        Cout,
        Kd,
        Kh,
        Kw,
        Dout,
        Hout,
        Wout,
        stride_xb,
        stride_xc,
        stride_xd,
        stride_xh,
        stride_xw,
        stride_wk,
        stride_wn,
        stride_ob,
        stride_oc,
        stride_od,
        stride_oh,
        stride_ow,
        M,
        N,
        K,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-optimized:
    3D convolution (valid, stride=1) + per-channel scale + tanh + per-channel scale + sigmoid,
    fused into a single high-performance kernel tuned for RTX 4090.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()

        if isinstance(kernel_size, int):
            Kd = Kh = Kw = kernel_size
        else:
            Kd, Kh, Kw = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, Kd, Kh, Kw)
        )
        self.conv_bias = nn.Parameter(torch.randn(out_channels))

        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return fused_conv3d_tanh_sigmoid(
            x,
            self.weight,
            self.conv_bias,
            self.scaling_factor,
            self.bias,
        )
