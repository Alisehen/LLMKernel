import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def sum_hw_kernel(
    x_ptr, out_ptr,
    B, C, HW,
    stride_b, stride_c,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    # pointer to (b, c, 0, 0)
    base_offset = b * stride_b + c * stride_c
    base_ptr = x_ptr + base_offset

    acc = 0.0
    offs = tl.arange(0, BLOCK_HW)
    for start in range(0, HW, BLOCK_HW):
        idx = start + offs
        mask = idx < HW
        vals = tl.load(base_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    tl.store(out_ptr + pid, acc)


@triton.jit
def gemm_mean_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # scale and add bias
    acc = acc * scale
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def row_logsumexp_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid  # 0 <= row < M

    row_ptr = x_ptr + row * stride_xm
    offs_n = tl.arange(0, BLOCK_N)

    # First pass: compute max over channels
    max_val = -float("inf")
    for start in range(0, N, BLOCK_N):
        cols = start + offs_n
        mask = cols < N
        vals = tl.load(
            row_ptr + cols * stride_xn,
            mask=mask,
            other=-float("inf"),
        )
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Second pass: compute sum(exp(x - max))
    sum_exp = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + offs_n
        mask = cols < N
        vals = tl.load(
            row_ptr + cols * stride_xn,
            mask=mask,
            other=-float("inf"),
        )
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)

    out = tl.log(sum_exp) + max_val
    out = out * 10.0
    tl.store(out_ptr + row, out)


def triton_sum_hw(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, C, H, W] (NCHW, contiguous)
    returns: [B, C] with sum over H,W
    """
    x = x.contiguous()
    B, C, H, W = x.shape
    HW = H * W
    out = torch.empty(B * C, device=x.device, dtype=x.dtype)

    stride_b = x.stride(0)
    stride_c = x.stride(1)

    grid = (B * C,)
    sum_hw_kernel[grid](
        x, out,
        B, C, HW,
        stride_b, stride_c,
        BLOCK_HW=256,
    )
    return out.view(B, C)


def triton_gemm_mean_bias(
    a: torch.Tensor,          # [B, C_in] = sum over H,W of input
    weight_sum: torch.Tensor, # [C_in, C_out] = sum over kh,kw of weight
    conv_bias: torch.Tensor,  # [C_out]
    bias2: torch.Tensor,      # [C_out, 1, 1]
    H: int, W: int,
    kH: int, kW: int,
) -> torch.Tensor:
    """
    Computes:
        g[b, c_out] = mean_HW(convT(x))[b, c_out] + bias2[c_out]
    using the algebraic reduction:
        mean_HW(convT) = (sum_HW(x) @ weight_sum) / (H_out*W_out) + conv_bias
    """
    B, C_in = a.shape
    C_in_w, C_out = weight_sum.shape
    assert C_in_w == C_in

    H_out = H + kH - 1
    W_out = W + kW - 1
    scale = 1.0 / float(H_out * W_out)

    # Bias after pooling (conv bias) + separate bias parameter
    bias2_flat = bias2.view(-1)
    bias_total = conv_bias + bias2_flat

    a = a.contiguous()
    w = weight_sum.contiguous()
    bias_total = bias_total.contiguous()

    g = torch.empty((B, C_out), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    gemm_mean_bias_kernel[grid](
        a, w, bias_total, g,
        B, C_out, C_in,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        g.stride(0), g.stride(1),
        scale,
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
    )
    return g


def triton_logsumexp_10(g: torch.Tensor) -> torch.Tensor:
    """
    g: [B, C_out]
    returns: [B, 1] = 10 * logsumexp(g, dim=1, keepdim=True)
    """
    g = g.contiguous()
    B, C_out = g.shape
    out = torch.empty(B, device=g.device, dtype=g.dtype)

    grid = (B,)
    row_logsumexp_kernel[grid](
        g, out,
        B, C_out,
        g.stride(0), g.stride(1),
        BLOCK_N=128,
    )
    return out.view(B, 1)


class ModelNew(nn.Module):
    """
    Triton-optimized version of the original model.

    Uses algebraic simplification of ConvTranspose2d + global average pooling:
    Instead of forming the full [B, C_out, H_out, W_out] tensor, it computes
    the pooled values directly from sums of inputs and kernel weights.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kH = kH
        self.kW = kW

        # ConvTranspose2d-like parameters: [in_channels, out_channels, kH, kW]
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kH, kW)
        )
        self.conv_bias = nn.Parameter(torch.randn(out_channels))

        # Additional bias added after global average pooling
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        B, C_in, H, W = x.shape
        assert C_in == self.in_channels

        # 1) Sum over spatial dims (H, W)
        sum_x = triton_sum_hw(x)  # [B, C_in]

        # 2) Sum weights over kernel dims (kh, kw): [C_in, C_out]
        weight_sum = self.weight.sum(dim=(2, 3))

        # 3) Compute global-average-pooled conv-transpose output + bias
        g = triton_gemm_mean_bias(
            sum_x,
            weight_sum,
            self.conv_bias,
            self.bias,
            H, W,
            self.kH, self.kW,
        )  # [B, C_out]

        # 4) logsumexp over channels and multiply by 10
        out = triton_logsumexp_10(g)  # [B, 1]
        return out
