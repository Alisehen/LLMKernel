# <optimized Triton code>

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles, high occupancy
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=2,
        ),
        # Larger N tile – better reuse of weights
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 16},
            num_warps=4,
            num_stages=2,
        ),
        # More compute per tile
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 16},
            num_warps=8,
            num_stages=3,
        ),
        # Deeper K tiles – better memory pipelining
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["N", "L_out", "Cin", "Cout", "K"],
)
@triton.jit
def conv1d_fwd_kernel(
    x_ptr,  # (N, Cin, L_in)
    w_ptr,  # (Cout, Cin, K)
    b_ptr,  # (Cout,) or dummy
    y_ptr,  # (N, Cout, L_out)
    N,
    L_in,
    L_out,
    stride,
    dilation,
    x_bs,
    x_cs,
    x_ls,
    w_os,
    w_cs,
    w_ks,
    y_bs,
    y_cs,
    y_ls,
    has_bias,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Flatten batch & spatial into M dimension
    M = N * L_out
    Ktot = Cin * K

    # ----- Program id mapping with GROUP_M for better weight cache reuse -----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(Cout, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    pid_m = first_pid_m + (pid % GROUP_M)
    pid_n = (pid % num_pid_in_group) // GROUP_M

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < Cout

    # Decode offs_m → (n_idx, l_out_idx)
    n_idx = offs_m // L_out
    l_out_idx = offs_m % L_out

    # Hints for compiler (help vectorization / address math)
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Broadcast indices for row/col
    n_b = n_idx[:, None]          # [BM, 1]
    l_out_b = l_out_idx[:, None]  # [BM, 1]

    # Reduction over Ktot = Cin * K
    for k0 in range(0, Ktot, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < Ktot

        ci = offs_k // K  # [BK]
        kw = offs_k % K   # [BK]

        ci_row = ci[None, :]      # [1, BK]
        kw_row = kw[None, :]      # [1, BK]

        # Compute input positions: l_in = l_out*stride + kw*dilation
        l_in = l_out_b * stride + kw_row * dilation  # [BM, BK]
        in_bounds = (l_in >= 0) & (l_in < L_in)
        in_mask = (mask_m[:, None]) & (mask_k[None, :]) & in_bounds

        # Input: x[n, ci, l_in]
        x_ptrs = x_ptr + n_b * x_bs + ci_row * x_cs + l_in * x_ls
        x_tile = tl.load(x_ptrs, mask=in_mask, other=0.0)

        # Weights: w[co, ci, kw]
        ci_col = ci[:, None]  # [BK, 1]
        kw_col = kw[:, None]  # [BK, 1]
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * w_os
            + ci_col * w_cs
            + kw_col * w_ks
        )
        w_mask = (mask_k[:, None]) & (mask_n[None, :])
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # GEMM accumulate (uses tensor cores for fp16/bf16 inputs)
        acc += tl.dot(x_tile, w_tile)

    # Bias add
    if has_bias != 0:
        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias_vals[None, :]

    # Store output: y[n, co, l_out]
    y_ptrs = (
        y_ptr
        + n_b * y_bs
        + offs_n[None, :] * y_cs
        + l_out_b * y_ls
    )
    out_mask = (mask_m[:, None]) & (mask_n[None, :])
    tl.store(y_ptrs, acc, mask=out_mask)


def conv1d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int,
    dilation: int,
) -> torch.Tensor:
    # CPU / non-CUDA fallback
    if not x.is_cuda:
        return torch.nn.functional.conv1d(
            x, weight, bias, stride=stride, dilation=dilation
        )

    assert x.ndim == 3
    assert weight.ndim == 3
    N, Cin, L_in = x.shape
    Cout, Cin_w, K = weight.shape
    assert Cin == Cin_w

    padding = 0
    L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    assert L_out > 0

    x_ = x.contiguous()
    w_ = weight.contiguous()
    y = torch.empty((N, Cout, L_out), device=x.device, dtype=x.dtype)

    x_bs, x_cs, x_ls = x_.stride()
    w_os, w_cs, w_ks = w_.stride()
    y_bs, y_cs, y_ls = y.stride()

    has_bias = 1 if bias is not None else 0
    if bias is not None:
        b_ = bias.contiguous()
    else:
        b_ = torch.empty(1, device=x.device, dtype=x.dtype)

    M = N * L_out

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        num_pid_m = triton.cdiv(M, BM)
        num_pid_n = triton.cdiv(Cout, BN)
        # 1D launch; GROUP_M controls scheduling order inside the kernel
        return (num_pid_m * num_pid_n,)

    conv1d_fwd_kernel[grid](
        x_,
        w_,
        b_,
        y,
        N,
        L_in,
        L_out,
        stride,
        dilation,
        x_bs,
        x_cs,
        x_ls,
        w_os,
        w_cs,
        w_ks,
        y_bs,
        y_cs,
        y_ls,
        has_bias,
        Cin=Cin,
        Cout=Cout,
        K=K,
    )

    return y


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return self.conv1d(x)

        weight = self.conv1d.weight
        bias = self.conv1d.bias
        stride = self.conv1d.stride[0]
        dilation = self.conv1d.dilation[0]

        return conv1d_triton(x, weight, bias, stride, dilation)
