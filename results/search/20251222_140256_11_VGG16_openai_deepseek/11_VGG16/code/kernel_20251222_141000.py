import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Extra Triton math helpers (manual implementations of commonly-missing ops)
# -----------------------------------------------------------------------------

@triton.jit
def tl_sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def tl_tanh(x):
    e_pos = tl.exp(x)
    e_neg = tl.exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)


@triton.jit
def tl_gelu(x):
    # Approximate GELU (tanh-based)
    k = 0.7978845608028654  # sqrt(2 / pi)
    c = 0.044715
    return 0.5 * x * (1.0 + tl_tanh(k * (x + c * x * x * x)))


@triton.jit
def tl_silu(x):
    return x * tl_sigmoid(x)


@triton.jit
def tl_mish(x):
    # mish(x) = x * tanh(softplus(x)), softplus(x)=log(1+exp(x))
    sp = tl.log(1.0 + tl.exp(x))
    return x * tl_tanh(sp)


@triton.jit
def tl_softmax_row(x):
    # Softmax along 1D row
    x_max = tl.max(x, axis=0)
    x_stable = x - x_max
    num = tl.exp(x_stable)
    den = tl.sum(num, axis=0)
    return num / den


# -----------------------------------------------------------------------------
# Optimized Triton Linear (Matmul + Bias) kernel
# - Fused: matmul + bias (no intermediate stores)
# - 1D grouped tiling for better L2 reuse (GROUP_M)
# - Autotuned for RTX 4090 with register-aware configs
# -----------------------------------------------------------------------------


@triton.autotune(
    configs=[
        # High-throughput tile, more warps, good for large, square-ish GEMMs
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=8,
            num_stages=2,
        ),
        # Rectangular tiles to handle tall matrices efficiently
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
        # Conservative baseline: smaller tile, lower register pressure
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 4,
            },
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def linear_kernel(
    x_ptr,  # [M, K]
    w_ptr,  # [K, N] = weight.T (K major)
    b_ptr,  # [N]
    y_ptr,  # [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # 1D program id -> 2D (pid_m, pid_n) with grouped M to improve L2 locality
    # -------------------------------------------------------------------------
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    group_pid = pid % num_pid_in_group

    first_pid_m = group_id * GROUP_M
    pid_m = first_pid_m + (group_pid % GROUP_M)
    pid_n = group_pid // GROUP_M

    # Even if pid_m >= num_pid_m, we simply mask all accesses out later.
    # This avoids divergent control-flow / early returns.

    # -------------------------------------------------------------------------
    # Compute tile offsets
    # -------------------------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_init = tl.arange(0, BLOCK_K)

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k_init, BLOCK_K)

    # Accumulator in FP32 for numerical stability and tensor-core throughput
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -------------------------------------------------------------------------
    # Main K loop: accumulate X @ W into acc
    #   X tile: [BLOCK_M, BLOCK_K]
    #   W tile: [BLOCK_K, BLOCK_N]
    # -------------------------------------------------------------------------
    k0 = 0
    while k0 < K:
        offs_k = k0 + offs_k_init

        # Masks for valid bounds
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_k = offs_k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        x_mask = mask_m[:, None] & mask_k[None, :]
        w_mask = mask_k[:, None] & mask_n[None, :]

        # Coalesced loads
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # FMA; allow_tf32 for TensorCore acceleration on 4090
        acc += tl.dot(x, w, allow_tf32=True)

        k0 += BLOCK_K

    # -------------------------------------------------------------------------
    # Bias add (fused; no intermediate stores)
    # -------------------------------------------------------------------------
    mask_n = offs_n < N
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b[None, :]

    # -------------------------------------------------------------------------
    # Final store (SINGLE tl.store as required)
    # -------------------------------------------------------------------------
    mask_m = offs_m < M
    out_mask = mask_m[:, None] & mask_n[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=out_mask)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [N, K]  (same layout as nn.Linear.weight)
    bias:   [N]
    returns y = x @ weight.T + bias
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA device"
    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, "Incompatible shapes for x and weight"

    # Ensure contiguous memory for coalesced access
    x_contig = x.contiguous()
    # Kernel expects [K, N] layout for W
    w_t = weight.t().contiguous()
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        BM = meta["BLOCK_M"]
        BN = meta["BLOCK_N"]
        num_pid_m = triton.cdiv(M, BM)
        num_pid_n = triton.cdiv(N, BN)
        # 1D launch over all (pid_m, pid_n) tiles
        return (max(1, num_pid_m * num_pid_n),)

    linear_kernel[grid](
        x_contig,
        w_t,
        bias,
        y,
        M,
        N,
        K,
        x_contig.stride(0),
        x_contig.stride(1),
        w_t.stride(0),
        w_t.stride(1),
        y.stride(0),
        y.stride(1),
    )
    return y


class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            # Avoid branching in the kernel; keep interface uniform
            b = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
        else:
            b = self.bias
        return triton_linear(x, self.weight, b)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        VGG16-style model with Triton-accelerated linear layers.
        """
        super(ModelNew, self).__init__()

        # Convolutional feature extractor (cuDNN-optimized)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected classifier with TritonLinear
        self.classifier = nn.Sequential(
            TritonLinear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            TritonLinear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            TritonLinear(4096, num_classes),
        )

    def forward(self, x):
        """
        x:  (batch_size, 3, 224, 224)
        returns: (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
