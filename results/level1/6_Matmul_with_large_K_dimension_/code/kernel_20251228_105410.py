import torch, torch.nn as nn, triton, triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,  # power-of-2
    BLOCK_N: tl.constexpr,  # power-of-2
    BLOCK_K: tl.constexpr,  # power-of-2
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m_broadcast = offs_m[:, None]
    offs_n_broadcast = offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_range = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + k_range

        a_ptrs = A_ptr + offs_m_broadcast * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n_broadcast * stride_bn

        a_mask = (offs_m_broadcast < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n_broadcast < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    c_ptrs = C_ptr + offs_m_broadcast * stride_cm + offs_n_broadcast * stride_cn
    c_mask = (offs_m_broadcast < M) & (offs_n_broadcast < N)

    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    High-performance Triton matmul: C = A @ B
    A: (M, K), B: (K, N)
    """
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D matrices"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, "Inner dimensions must match for matmul"

    orig_device = A.device
    # Compute on CUDA for Triton; move if necessary
    if orig_device.type != "cuda":
        A = A.to("cuda")
        B = B.to("cuda")
    else:
        # Ensure both on same CUDA device
        if B.device != orig_device:
            B = B.to(orig_device)

    A = A.contiguous()
    B = B.contiguous()

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )

    if orig_device.type != "cuda":
        C = C.to(orig_device)
    return C


class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension,
    implemented with a high-performance Triton kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return triton_matmul(A, B)
