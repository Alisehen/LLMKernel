import torch
import torch.nn as nn
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 4},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 2},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 2},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 2},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 2},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["N", "C_in", "C_out", "H", "W"],
)
@triton.jit
def conv1x1_fwd_kernel(
    x_ptr,  # *[N, C_in, H, W]
    w_ptr,  # *[C_out, C_in, 1, 1] -> [C_out, C_in]
    b_ptr,  # *[C_out] or dummy if HAS_BIAS == False
    y_ptr,  # *[N, C_out, H, W]
    N, C_in, H, W, C_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc,
    stride_yn, stride_yc, stride_yh, stride_yw,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Total M dimension = N * H * W
    M = N * H * W

    # -------------------------------------------------------------------------
    # Program id grouping along M to improve L2 reuse of weights (w_ptr)
    # -------------------------------------------------------------------------
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group

    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    # Guard against out-of-range tiles (can happen in last group)
    if pid_m >= num_pid_m:
        return

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < C_out

    # Decode flat M index -> (n_idx, oh, ow)
    ohw = H * W
    n_idx = offs_m // ohw
    hw_idx = offs_m % ohw
    oh = hw_idx // W
    ow = hw_idx % W

    # Broadcast shapes
    n_idx = n_idx[:, None]  # [BM, 1]
    oh = oh[:, None]        # [BM, 1]
    ow = ow[:, None]        # [BM, 1]
    offs_n_b = offs_n[None, :]  # [1, BN]

    # Precompute base pointers for x and y (independent of K)
    base_x = (
        x_ptr
        + n_idx * stride_xn
        + oh * stride_xh
        + ow * stride_xw
    )  # [BM, 1]
    base_y = (
        y_ptr
        + n_idx * stride_yn
        + oh * stride_yh
        + ow * stride_yw
    )  # [BM, 1]

    # Precompute base pointer for w along output channels
    base_w = w_ptr + offs_n_b * stride_wn  # [1, BN]

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over input channels C_in
    for k0 in range(0, C_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]
        mask_k = offs_k < C_in

        offs_k_col = offs_k[None, :]  # [1, BK]   (for x)
        offs_k_row = offs_k[:, None]  # [BK, 1]   (for w)

        tl.multiple_of(offs_k, BLOCK_K)

        # x[n_idx, offs_k, oh, ow] -> [BM, BK]
        ptrs_x = base_x + offs_k_col * stride_xc
        # w[offs_n, offs_k] -> [BK, BN]
        ptrs_w = base_w + offs_k_row * stride_wc

        mask_x = mask_m[:, None] & mask_k[None, :]
        mask_w = mask_k[:, None] & mask_n[None, :]

        a = tl.load(ptrs_x, mask=mask_x, other=0.0)
        b = tl.load(ptrs_w, mask=mask_w, other=0.0)

        # GEMM tile: [BM, BK] x [BK, BN] -> [BM, BN]
        acc += tl.dot(a, b)

    # Optional bias add
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        bias_vals = bias_vals.to(tl.float32)
        acc += bias_vals[None, :]

    # Store output y[n_idx, offs_n, oh, ow]
    ptrs_y = base_y + offs_n_b * stride_yc
    mask_y = mask_m[:, None] & mask_n[None, :]
    tl.store(ptrs_y, acc, mask=mask_y)


def triton_pointwise_conv2d_1x1(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pointwise 1x1 Conv2d implemented as a fused GEMM over (N*H*W, C_in) x (C_in, C_out).
    Uses an autotuned Triton kernel optimized for Ada GPUs.
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernel."
    assert weight.is_cuda, "Weight must be on CUDA for Triton kernel."
    assert x.ndim == 4
    assert weight.ndim == 4

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    assert weight.shape[1] == C_in
    assert weight.shape[2] == 1 and weight.shape[3] == 1

    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)

    M = N * H * W

    def grid(meta):
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        GROUP_M = meta["GROUP_M"]

        num_pid_m = triton.cdiv(M, BLOCK_M)
        num_pid_n = triton.cdiv(C_out, BLOCK_N)
        return (triton.cdiv(num_pid_m, GROUP_M) * num_pid_n,)

    has_bias = bias is not None
    b_ptr = bias if has_bias else weight  # dummy pointer when no bias

    conv1x1_fwd_kernel[grid](
        x,
        weight,
        b_ptr,
        y,
        N,
        C_in,
        H,
        W,
        C_out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        HAS_BIAS=has_bias,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated pointwise 1x1 Conv2d replacement.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return self.conv(x)

        try:
            return triton_pointwise_conv2d_1x1(x, self.conv.weight, self.conv.bias)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg:
                raise

            x_cpu = x.detach().cpu()
            w_cpu = self.conv.weight.detach().cpu()
            b_cpu = self.conv.bias.detach().cpu() if self.conv.bias is not None else None
            y_cpu = F.conv2d(x_cpu, w_cpu, b_cpu, stride=1, padding=0)
            return y_cpu
