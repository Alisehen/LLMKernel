# <corrected code>
import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Triton Kernels
# -----------------------------------------------------------------------------


@triton.autotune(
    configs=[
        # Balanced tiles for general GEMM
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
        # Larger square tiles to better use tensor cores on big problems
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 8}, num_stages=3),
        # More N-heavy for MLP second layer (K relatively small vs N)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_warps': 8}, num_stages=3),
        # Small-M friendly configs (for batched small batches)
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'num_warps': 4}, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_kernel(
    a_ptr,        # *f32, [M, K]
    b_ptr,        # *f32, [K, N]
    bias_ptr,     # *f32, [N]
    c_ptr,        # *f32, [M, N]
    M, N, K,      # int32
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids for 2D tiling over M, N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Alignment hints
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    tl.multiple_of(offs_k, BLOCK_K)

    # Initialize accumulators in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointers for the first K tile
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k

        # Masks for boundary handling on the last K tile
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Fused matmul accumulate
        acc += tl.dot(a, b, allow_tf32=True)

        # Advance block pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused bias add
    n_mask = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# -----------------------------------------------------------------------------
# Wrapper Functions
# -----------------------------------------------------------------------------


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      [M, K]
    weight: [K, N]   (already transposed weight matrix, contiguous)
    bias:   [N]
    returns [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"
    assert x.dtype == weight.dtype == bias.dtype == torch.float32, "Kernel assumes float32 tensors"

    M, K = x.shape
    K_w, N = weight.shape
    assert K_w == K, f"Incompatible shapes: x [{M}, {K}], weight [{K_w}, {N}]"

    # Ensure contiguous for predictable strides and better codegen
    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        # 2D launch grid: one program per (BLOCK_M x BLOCK_N) output tile
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    linear_bias_kernel[grid](
        x_contig, w_contig, b_contig, y,
        M, N, K,
        x_contig.stride(0), x_contig.stride(1),
        w_contig.stride(0), w_contig.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


# -----------------------------------------------------------------------------
# Model with Triton-accelerated Linears
# -----------------------------------------------------------------------------


class ModelNew(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        """
        Vision Transformer (ViT) with Triton-accelerated linear layers
        for patch embedding and MLP head.
        """
        super(ModelNew, self).__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim

        # Positional embedding and CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Patch embedding linear (implemented via Triton)
        patch_linear = nn.Linear(patch_dim, dim)
        # Store weights transposed and contiguous for GEMM: [K, N] = [patch_dim, dim]
        self.patch_weight = nn.Parameter(patch_linear.weight.detach().t().contiguous())
        self.patch_bias = nn.Parameter(patch_linear.bias.detach().clone())

        # Transformer encoder (PyTorch implementation)
        # Use batch_first=False to match the reference implementation
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=False,
            ),
            num_layers=depth,
        )

        self.to_cls_token = nn.Identity()

        # MLP head: two Triton linear layers with GELU and dropout in between.
        mlp_fc1 = nn.Linear(dim, mlp_dim)
        self.mlp_w1 = nn.Parameter(mlp_fc1.weight.detach().t().contiguous())  # [dim, mlp_dim]
        self.mlp_b1 = nn.Parameter(mlp_fc1.bias.detach().clone())

        self.mlp_dropout = nn.Dropout(dropout)

        mlp_fc2 = nn.Linear(mlp_dim, num_classes)
        self.mlp_w2 = nn.Parameter(mlp_fc2.weight.detach().t().contiguous())  # [mlp_dim, num_classes]
        self.mlp_b2 = nn.Parameter(mlp_fc2.bias.detach().clone())

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [batch_size, channels, image_size, image_size]
        returns: [batch_size, num_classes]
        """
        p = self.patch_size
        bsz = img.shape[0]

        # Patch extraction: [B, C, H, W] -> [B, num_patches, patch_dim]
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(
            img.shape[0], -1, p * p * img.shape[1]
        )

        # Patch embedding via Triton GEMM
        B, N, D_in = x.shape  # N: num_patches, D_in: patch_dim
        x_flat = x.reshape(B * N, D_in)
        x_emb = triton_linear(x_flat, self.patch_weight, self.patch_bias)  # [B*N, dim]
        x = x_emb.reshape(B, N, self.dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(bsz, -1, -1)  # [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, dim]

        # Add positional embeddings and apply dropout
        x = x + self.pos_embedding  # [1, N+1, dim] broadcasts over batch
        x = self.dropout(x)

        # Transformer encoder (batch_first=False to match reference)
        # Convert from [B, S, D] to [S, B, D]
        x = x.transpose(0, 1)  # [S, B, D]
        x = self.transformer(x)

        # Take CLS token (first in sequence dimension)
        x = self.to_cls_token(x[0])  # [B, D]

        # MLP head: Triton Linear -> GELU -> Dropout -> Triton Linear
        x = triton_linear(x, self.mlp_w1, self.mlp_b1)
        x = torch.nn.functional.gelu(x)
        x = self.mlp_dropout(x)
        x = triton_linear(x, self.mlp_w2, self.mlp_b2)

        return x
