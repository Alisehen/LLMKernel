import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_fwd_kernel(
    x_ptr,  # (N, Cin, L_in)
    w_ptr,  # (Cout, Cin, K)
    b_ptr,  # (Cout,) or dummy
    y_ptr,  # (N, Cout, L_out)
    N,      # batch size
    L_in,   # input length
    L_out,  # output length
    stride,
    dilation,
    x_bs, x_cs, x_ls,
    w_os, w_cs, w_ks,
    y_bs, y_cs, y_ls,
    has_bias,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implicit GEMM 1D convolution: treats output as (M = N*L_out, Cout),
    reduction dimension Ktot = Cin * K.
    """

    # Total rows (output positions over batch & length)
    M = N * L_out
    Ktot = Cin * K

    pid_m = tl.program_id(axis=0)  # tile over M dimension
    pid_n = tl.program_id(axis=1)  # tile over Cout dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = offs_m < M
    mask_n = offs_n < Cout

    # Decode offs_m → (n, l_out)
    n_idx = offs_m // L_out           # [BLOCK_M]
    l_out_idx = offs_m % L_out        # [BLOCK_M]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over Ktot = Cin * K in tiles of BLOCK_K
    for k0 in range(0, Ktot, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = offs_k < Ktot

        ci = offs_k // K  # input channel index, [BLOCK_K]
        kw = offs_k % K   # kernel position,      [BLOCK_K]

        # Broadcast indices
        n_b = n_idx[:, None]          # [BM, 1]
        l_out_b = l_out_idx[:, None]  # [BM, 1]
        ci_row = ci[None, :]          # [1, BK]
        kw_row = kw[None, :]          # [1, BK]

        # Compute input positions l_in = l_out*stride + kw*dilation
        l_in = l_out_b * stride + kw_row * dilation  # [BM, BK]

        # Bounds mask for input
        in_bounds = (l_in >= 0) & (l_in < L_in)
        in_mask = (mask_m[:, None]) & (mask_k[None, :]) & in_bounds  # [BM, BK]

        # Input pointers: x[n, ci, l_in]
        x_ptrs = (
            x_ptr
            + n_b * x_bs
            + ci_row * x_cs
            + l_in * x_ls
        )
        x_tile = tl.load(x_ptrs, mask=in_mask, other=0.0)  # [BM, BK]

        # Weight pointers: w[co, ci, kw]
        ci_col = ci[:, None]  # [BK, 1]
        kw_col = kw[:, None]  # [BK, 1]
        w_ptrs = (
            w_ptr
            + offs_n[None, :] * w_os
            + ci_col * w_cs
            + kw_col * w_ks
        )  # [BK, BN]
        w_mask = (mask_k[:, None]) & (mask_n[None, :])
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BK, BN]

        # GEMM accumulation
        acc += tl.dot(x_tile, w_tile)  # [BM, BN]

    # Add bias if present
    if has_bias != 0:
        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)  # [BN]
        acc = acc + bias_vals[None, :]  # broadcast add

    # Store output: y[n, co, l_out]
    n_b = n_idx[:, None]         # [BM, 1]
    l_out_b = l_out_idx[:, None] # [BM, 1]
    y_ptrs = (
        y_ptr
        + n_b * y_bs
        + offs_n[None, :] * y_cs
        + l_out_b * y_ls
    )
    out_mask = (mask_m[:, None]) & (mask_n[None, :])
    tl.store(y_ptrs, acc, mask=out_mask)


def conv1d_triton(x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor,
                  stride: int,
                  dilation: int) -> torch.Tensor:
    """
    x:      (N, Cin, L_in)
    weight: (Cout, Cin, K)
    bias:   (Cout,) or None
    """
    # CPU fallback for safety
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

    x_bs, x_cs, x_ls = x_.stride()  # (N, Cin, L_in)
    w_os, w_cs, w_ks = w_.stride()  # (Cout, Cin, K)
    y_bs, y_cs, y_ls = y.stride()   # (N, Cout, L_out)

    has_bias = 1 if bias is not None else 0
    if bias is not None:
        b_ = bias.contiguous()
    else:
        # Dummy tensor; won't be read when has_bias == 0
        b_ = torch.empty(1, device=x.device, dtype=x.dtype)

    # Tile sizes (powers of 2, constexpr)
    BLOCK_M = 64  # rows over N*L_out
    BLOCK_N = 64  # output channels
    BLOCK_K = 32  # reduction over Cin*K

    M = N * L_out

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(Cout, meta["BLOCK_N"]),
    )

    conv1d_fwd_kernel[grid](
        x_, w_, b_, y,
        N, L_in, L_out,
        stride, dilation,
        x_bs, x_cs, x_ls,
        w_os, w_cs, w_ks,
        y_bs, y_cs, y_ls,
        has_bias,
        Cin=Cin,
        Cout=Cout,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated replacement for the given Conv1d-based model.
    """

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
        # Use PyTorch to manage parameters; bypass its forward with our Triton kernel
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU / non-CUDA fallback
        if not x.is_cuda:
            return self.conv1d(x)

        weight = self.conv1d.weight
        bias = self.conv1d.bias
        stride = self.conv1d.stride[0]
        dilation = self.conv1d.dilation[0]

        return conv1d_triton(x, weight, bias, stride, dilation)
