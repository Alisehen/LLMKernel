import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ----------------------------- Conv2D + HSwish kernel -----------------------------


@triton.autotune(
    configs=[
        # Conservative baseline: good for high register pressure / small problems
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # More rows in M dimension (N*H_out*W_out)
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Wider in C_out dimension, more warps for compute throughput
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        # Aggressive square tile for large problems on 4090
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["N", "C_out", "H_out", "W_out"],
)
@triton.jit
def conv2d_hswish_gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, H_in, W_in,
    C_out, H_out, W_out,
    subtract_value,
    # constexpr
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    C_IN: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
):
    """
    GEMM-style implicit-im2col conv2d fused with:
      bias add, subtract_value, and HardSwish activation.

    Input:  x_ptr  [N, C_IN, H_in, W_in]  (NCHW, contiguous)
    Weight: w_ptr  [K_TOTAL, C_out] where K_TOTAL = C_IN * K_H * K_W (row-major)
    Bias:   b_ptr  [C_out]
    Output: y_ptr  [N, C_out, H_out, W_out]

    Grid over: [M = N*H_out*W_out, C_out].
    Single tl.store: final output only.
    """

    # ------------------------- Tile coordinates -------------------------
    pid_m = tl.program_id(0)  # rows (M = N*H_out*W_out)
    pid_n = tl.program_id(1)  # cols (C_out)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    HW_out = H_out * W_out
    P = N * HW_out  # total output rows

    mask_m = offs_m < P
    mask_n = offs_n < C_out

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # ------------------------- Decode m -> (n, oh, ow) -------------------------
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Broadcasted indices for [BM, BK] tiles
    n_b = n_idx[:, None]   # [BM,1]
    oh_b = oh_idx[:, None] # [BM,1]
    ow_b = ow_idx[:, None] # [BM,1]

    # Precompute factor that does not depend on K-tile:
    # base_n = n * C_IN * H_in
    base_n = (n_idx * (C_IN * H_in))[:, None]  # [BM,1]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Total reduction dimension (compile-time constant)
    K_TOTAL = C_IN * K_H * K_W

    # ------------------------- Reduction over K -------------------------
    # Fully unrolled at compile-time for given C_IN, K_H, K_W, BLOCK_K
    for k0 in range(0, K_TOTAL, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]
        mask_k = offs_k < K_TOTAL
        tl.multiple_of(offs_k, BLOCK_K)

        # Decode K index -> (ic, kh, kw)
        kk = offs_k
        ic = kk // (K_H * K_W)
        rem_k = kk % (K_H * K_W)
        kh = rem_k // K_W
        kw = rem_k % K_W

        ic_b = ic[None, :]  # [1,BK]
        kh_b = kh[None, :]
        kw_b = kw[None, :]

        # Spatial positions for this K-tile [BM,BK]
        ih = oh_b + kh_b  # [BM,BK]
        iw = ow_b + kw_b  # [BM,BK]

        # Compute flattened input index:
        # idx = (((n*C_IN + ic)*H_in + ih)*W_in + iw)
        # base_n:  n*C_IN*H_in (precomputed)  -> [BM,1]
        # ic_h:    ic*H_in                   -> [1,BK]
        ic_h = (ic * H_in)[None, :]     # [1,BK]
        base_nc_h = base_n + ic_h       # [BM,BK]
        tmp_in = base_nc_h + ih         # [BM,BK]
        in_index = tmp_in * W_in + iw   # [BM,BK]

        a_mask = mask_m[:, None] & mask_k[None, :]
        # Load input tile (promote to fp32 for compute)
        a = tl.load(x_ptr + in_index, mask=a_mask, other=0.0).to(tl.float32)  # [BM,BK]

        # B tile: w_ptr is [K_TOTAL, C_out] row-major
        b_index = offs_k[:, None] * C_out + offs_n[None, :]  # [BK,BN]
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(w_ptr + b_index, mask=b_mask, other=0.0).to(tl.float32)  # [BK,BN]

        # Matmul accumulate (tf32 enabled on 4090)
        acc += tl.dot(a, b, allow_tf32=True)

    # ------------------------- Fused epilogue -------------------------
    # Bias: broadcast over rows
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)  # [BN]
    acc += bias_vals[None, :]  # broadcast along M

    # Subtract scalar (in fp32)
    acc = acc - subtract_value

    # HardSwish: x * clamp(x+3, 0, 6) / 6  (in-register, no intermediate store)
    tmp = acc + 3.0
    tmp = tl.maximum(tmp, 0.0)
    tmp = tl.minimum(tmp, 6.0)
    acc = acc * tmp * (1.0 / 6.0)

    # ------------------------- Store -------------------------
    # Decode back to NCHW layout
    n2 = n_idx[:, None]   # [BM,1]
    oh2 = oh_idx[:, None] # [BM,1]
    ow2 = ow_idx[:, None] # [BM,1]
    oc2 = offs_n[None, :] # [1,BN]

    out_index = (((n2 * C_out + oc2) * H_out + oh2) * W_out + ow2)  # [BM,BN]
    out_mask = mask_m[:, None] & mask_n[None, :]

    # Single final store for fused Conv + HSwish
    tl.store(y_ptr + out_index, acc, mask=out_mask)


def conv2d_hswish(x, weight, bias, subtract_value: float):
    """
    NCHW conv2d (stride=1, padding=0, dilation=1) fused with:
      bias add, subtract_value, and HardSwish activation.
    Implemented as GEMM with implicit im2col.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in == C_in_w

    # For stride=1, padding=0
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Pre-pack weight to [K_total, C_out] row-major for coalesced loads
    K_total = C_in * K_H * K_W
    weight_2d = weight.view(C_out, K_total).transpose(0, 1).contiguous()

    P = N * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_hswish_gemm_kernel[grid](
        x, weight_2d, bias, y,
        N, H_in, W_in,
        C_out, H_out, W_out,
        float(subtract_value),
        C_IN=C_in, K_H=K_H, K_W=K_W,
    )

    return y


# ----------------------------- MaxPool2D + Mish kernel -----------------------------


@triton.autotune(
    configs=[
        # Baseline: balanced tile
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        ),
        # More rows (N*H_out*W_out)
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        ),
        # Aggressive wide tile in C dimension
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["N", "C", "H_out", "W_out"],
)
@triton.jit
def maxpool2d_mish_kernel(
    x_ptr, y_ptr,
    N, C, H_in, W_in,
    H_out, W_out,
    # constexpr
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    K_P: tl.constexpr,
):
    """
    NCHW MaxPool2d with kernel_size=stride=K_P, padding=0, dilation=1,
    fused with Mish activation.

    Grid covers output [M = N*H_out*W_out, C].
    Single tl.store: final MaxPool + Mish output.
    """
    pid_m = tl.program_id(0)  # flattened (N * H_out * W_out)
    pid_n = tl.program_id(1)  # channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    P = N * H_out * W_out
    mask_m = offs_m < P
    mask_n = offs_n < C

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    n = n_idx[:, None]   # [BM,1]
    oh = oh_idx[:, None] # [BM,1]
    ow = ow_idx[:, None] # [BM,1]
    c = offs_n[None, :]  # [1,BN]

    full_mask = mask_m[:, None] & mask_n[None, :]

    # Precompute base offsets independent of kernel loop:
    # plane_base = (n*C + c) * H_in * W_in  -> starting index of full HxW plane
    plane_base = (n * C + c) * (H_in * W_in)  # [BM,BN]

    # Pooling window top-left (in input coordinates)
    oh_kp = oh * K_P  # [BM,1]
    ow_kp = ow * K_P  # [BM,1]

    # Initialize max accumulator in fp32
    neg_inf = -1e30
    acc = tl.full((BLOCK_M, BLOCK_N), neg_inf, dtype=tl.float32)

    # MaxPool2d over K_P x K_P with stride=K_P, no padding
    # Unroll over small K_P using tl.static_range for better ILP
    for kh in tl.static_range(0, K_P):
        ih = oh_kp + kh        # [BM,1]
        row_offset = ih * W_in # [BM,1]
        for kw in tl.static_range(0, K_P):
            iw = ow_kp + kw          # [BM,1]
            intra = row_offset + iw  # [BM,1]  offset within HxW plane
            in_index = plane_base + intra  # [BM,BN]
            x_vals = tl.load(
                x_ptr + in_index,
                mask=full_mask,
                other=neg_inf,
            ).to(tl.float32)
            acc = tl.maximum(acc, x_vals)

    # Avoid propagating -inf into transcendental functions for masked-out lanes
    acc = tl.where(full_mask, acc, 0.0)

    # Mish activation: x * tanh(softplus(x)), softplus(x) = log(1 + exp(x))
    # Implement tanh manually (tl.tanh is not available)
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    t = (e2 - 1.0) / (e2 + 1.0)
    out_vals = acc * t

    # Store using the same tile indices (single store)
    out_index = (((n * C + c) * H_out + oh) * W_out + ow)  # [BM,BN]
    tl.store(y_ptr + out_index, out_vals, mask=full_mask)


def maxpool2d_mish(x, kernel_size: int):
    """
    NCHW MaxPool2d with kernel_size=stride=kernel_size (no padding),
    fused with Mish activation.
    """
    assert x.is_cuda
    x = x.contiguous()

    N, C, H_in, W_in = x.shape
    K_P = int(kernel_size)

    H_out = H_in // K_P
    W_out = W_in // K_P

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    P = N * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C, meta["BLOCK_N"]),
        )

    maxpool2d_mish_kernel[grid](
        x, y,
        N, C, H_in, W_in,
        H_out, W_out,
        K_P=K_P,
    )

    return y


# ----------------------------- Model -----------------------------


class ModelNew(nn.Module):
    """
    Triton-optimized model:
      Conv2d -> subtract constant -> HardSwish -> MaxPool2d -> Mish
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        k = int(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = int(pool_kernel_size)

        # Parameters equivalent to nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, k, k)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Kaiming uniform initialization similar to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()

        device = x.device
        if not self.weight.is_cuda:
            self.weight.data = self.weight.data.to(device)
        if not self.bias.is_cuda:
            self.bias.data = self.bias.data.to(device)

        x = conv2d_hswish(x, self.weight, self.bias, self.subtract_value)
        x = maxpool2d_mish(x, self.pool_kernel_size)
        return x
