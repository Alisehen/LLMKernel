import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
            },
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
            },
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
            },
            num_warps=4,
            num_stages=4,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def int8_matmul_rowwise_dequant_kernel(
    a_ptr,        # int8  [M, K]
    b_ptr,        # int8  [K, N]  (weight_int8.T, contiguous)
    scale_x_ptr,  # fp32 [M]
    scale_w_ptr,  # fp32 [N]
    bias_ptr,     # fp16 [N]
    c_ptr,        # fp16 [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D tile coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for output tile (shared by all fused ops)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    # Base pointers for first K-tile of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Masks derived once from output coordinates
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulate in int32 to leverage INT8 Tensor Cores / DP4A
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # K-loop
    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k

        a_mask = mask_m[:, None] & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & mask_n[None, :]

        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)

        # int8 x int8 -> int32
        acc += tl.dot(a, b, out_dtype=tl.int32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Convert accumulation to fp32 for dequant + bias
    acc = tl.cast(acc, tl.float32)

    # Row-wise and column-wise scales, using same offsets
    scale_x = tl.load(scale_x_ptr + offs_m, mask=mask_m, other=0.0)  # [BLOCK_M]
    scale_w = tl.load(scale_w_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]

    # Dequantization factor
    divfactor = 1.0 / (127.0 * 127.0)

    # Fused scaling: (int32 dot) * (1/(127*127)) * scale_x[m] * scale_w[n]
    acc = acc * divfactor
    acc = acc * scale_x[:, None]
    acc = acc * scale_w[None, :]

    # Bias add (fp16 -> fp32), sharing same N-offsets
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)  # fp16
    bias_f32 = tl.cast(bias, tl.float32)
    acc = acc + bias_f32[None, :]

    # Store result as fp16
    c = tl.cast(acc, tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, c, mask=mask_out)


def fused_int8_rowwise_dequant_matmul(x, weight_int8, scale_x, scale_w, bias):
    """
    x:           int8  [M, K]
    weight_int8: int8  [N, K]  (original storage, we transpose here)
    scale_x:     fp32 [M]
    scale_w:     fp32 [N]
    bias:        fp16 [N]

    Computes:
        output = (x.float() @ weight_int8.t().float()) * scale_x * scale_w / (127*127) + bias
        output is returned as float16.
    """
    assert x.dtype == torch.int8
    assert weight_int8.dtype == torch.int8
    assert scale_x.dtype == torch.float32
    assert scale_w.dtype == torch.float32
    assert bias.dtype == torch.float16

    M, K = x.shape
    N = weight_int8.shape[0]

    # Prepare B = weight_int8.T [K, N], contiguous for coalesced loads along N
    b = weight_int8.t().contiguous()

    c = torch.empty((M, N), device=x.device, dtype=torch.float16)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    int8_matmul_rowwise_dequant_kernel[grid](
        x, b, scale_x, scale_w, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


class ModelNew(nn.Module):
    """
    Triton-optimized INT8 MatMul with Row-wise Dequantization.

    Semantics match the reference model:
        output = (x.float() @ weight_int8.t().float()) * scale_x * scale_w / (127*127) + bias
        output is returned as float16.
    """
    def __init__(self, in_features=2048, out_features=2048):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameters: INT8 weights, per-output scale & bias
        self.weight_int8 = nn.Parameter(
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8),
            requires_grad=False,
        )
        self.scale_w = nn.Parameter(
            torch.randn(out_features, dtype=torch.float32).abs() * 0.01,
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.randn(out_features, dtype=torch.float16) * 0.01,
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        return fused_int8_rowwise_dequant_matmul(
            x, self.weight_int8, scale_x, self.scale_w, self.bias
        )
