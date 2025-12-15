import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Conservative baseline (required): balanced tile, low stages
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=2,
        ),
        # Higher occupancy variant: same tile, more warps
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=8,
            num_stages=2,
        ),
        # More arithmetic intensity along N; still conservative stages
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def fused_gemm_bn_gelu_relu_kernel(
    a_ptr,              # *f32, (M, K)
    b_ptr,              # *f32, (K, N) = weight^T
    bias_lin_ptr,       # *f32, (N,)
    bn_weight_ptr,      # *f32, (N,)
    bn_bias_ptr,        # *f32, (N,)
    bn_mean_ptr,        # *f32, (N,)
    bn_var_ptr,         # *f32, (N,)
    c_ptr,              # *f32, (M, N)

    M, N, K,
    eps,                # f32, BN epsilon

    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # Program id / tile mapping with M-grouping for better L2 reuse of B
    # -------------------------------------------------------------------------
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Output tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Bounds masks
    mask_m = offs_m < M
    mask_n = offs_n < N

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)

    # -------------------------------------------------------------------------
    # GEMM: C = A @ B   (fp32 accum, TF32 tensor-cores where possible)
    # -------------------------------------------------------------------------
    offs_k = tl.arange(0, BLOCK_K)
    tl.multiple_of(offs_k, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Initial pointers for this K-tile
    a_ptrs = a_ptr + (
        offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    # We keep row/col masks that don't depend on K
    a_mask_base = mask_m[:, None]
    b_mask_base = mask_n[None, :]

    k = 0
    while k < K:
        k_offsets = k + offs_k
        k_mask = k_offsets < K

        a = tl.load(
            a_ptrs,
            mask=a_mask_base & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & b_mask_base,
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers to next K-slab
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # -------------------------------------------------------------------------
    # Fused epilogue:  BN + GELU + ReLU
    # y = ReLU(GELU(BN(acc + bias)))
    # -------------------------------------------------------------------------
    # Load per-channel parameters once per tile (broadcast along M)
    bias_lin = tl.load(
        bias_lin_ptr + offs_n,
        mask=mask_n,
        other=0.0,
    )[None, :]  # (1, BLOCK_N)

    bn_weight = tl.load(
        bn_weight_ptr + offs_n,
        mask=mask_n,
        other=1.0,
    )[None, :]
    bn_bias = tl.load(
        bn_bias_ptr + offs_n,
        mask=mask_n,
        other=0.0,
    )[None, :]
    bn_mean = tl.load(
        bn_mean_ptr + offs_n,
        mask=mask_n,
        other=0.0,
    )[None, :]
    bn_var = tl.load(
        bn_var_ptr + offs_n,
        mask=mask_n,
        other=1.0,
    )[None, :]

    # --- Optimized BatchNorm algebra ---
    # Original:
    #   y = (acc + bias - mean) * rsqrt(var+eps) * weight + bn_bias
    # Rewritten:
    #   scale = rsqrt(var+eps) * weight
    #   shift = (bias - mean) * scale + bn_bias
    #   y = acc * scale + shift
    # Fewer per-element ops and lower register pressure.
    var_eps = bn_var + eps
    inv_std = tl.rsqrt(var_eps)
    scale = inv_std * bn_weight
    shift = (bias_lin - bn_mean) * scale + bn_bias

    y = acc * scale + shift

    # --- GELU (erf approximation) ---
    sqrt_2_inv = 0.7071067811865476  # 1/sqrt(2)
    u = y * sqrt_2_inv

    sign = tl.where(u >= 0, 1.0, -1.0)
    absu = tl.abs(u)
    t = 1.0 / (1.0 + 0.3275911 * absu)

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    erf_u = sign * (1.0 - poly * tl.exp(-absu * absu))

    gelu = 0.5 * y * (1.0 + erf_u)

    # --- ReLU ---
    out = tl.maximum(gelu, 0.0)

    # Store
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, out, mask=out_mask)


def fused_gemm_bn_gelu_relu(x: torch.Tensor, linear: nn.Linear, bn: nn.BatchNorm1d):
    """
    Eval-mode fused path:
        y = ReLU(GELU(BN(x @ W^T + b)))
    Uses a single high-performance Triton kernel on CUDA.
    """
    assert linear.bias is not None, "Linear layer must have bias for this fused kernel."

    if x.device.type != "cuda":
        y = linear(x)
        y = bn(y)
        y = torch.nn.functional.gelu(y)
        y = torch.nn.functional.relu(y)
        return y

    M, K = x.shape
    N = linear.weight.shape[0]

    # Ensure explicit fp32 layout for the matmul
    # Keep conversions here to avoid type promotion inside the kernel.
    x_fp32 = x.to(device=x.device, dtype=torch.float32, non_blocking=True)
    W = linear.weight.to(device=x.device, dtype=torch.float32, non_blocking=True)

    # B is (K, N), row-major, for coalesced loads in Triton
    B = W.t().contiguous()

    # Output
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # BN parameters (eval mode uses running stats)
    bn_weight = bn.weight.to(device=x.device, dtype=torch.float32, non_blocking=True)
    bn_bias = bn.bias.to(device=x.device, dtype=torch.float32, non_blocking=True)
    bn_mean = bn.running_mean.to(device=x.device, dtype=torch.float32, non_blocking=True)
    bn_var = bn.running_var.to(device=x.device, dtype=torch.float32, non_blocking=True)
    eps = float(bn.eps)

    # Grid: 1D over (M, N) tiles; kernel maps pid -> (pid_m, pid_n)
    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )

    fused_gemm_bn_gelu_relu_kernel[grid](
        x_fp32,
        B,
        linear.bias.to(device=x.device, dtype=torch.float32, non_blocking=True),
        bn_weight,
        bn_bias,
        bn_mean,
        bn_var,
        y,
        M,
        N,
        K,
        eps,
        x_fp32.stride(0),
        x_fp32.stride(1),
        B.stride(0),
        B.stride(1),
        y.stride(0),
        y.stride(1),
    )

    return y


class ModelNew(nn.Module):
    """
    GEMM + BatchNorm + GELU + ReLU.
    Uses fused Triton kernel in eval mode on CUDA,
    and the standard PyTorch implementation otherwise.
    """

    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training mode or non-CUDA: full PyTorch path for correct semantics
        if self.training or x.device.type != "cuda":
            x = self.gemm(x)
            x = self.batch_norm(x)
            x = torch.nn.functional.gelu(x)
            x = torch.nn.functional.relu(x)
            return x

        # Eval-mode fast path: fused Triton kernel
        return fused_gemm_bn_gelu_relu(x, self.gemm, self.batch_norm)
