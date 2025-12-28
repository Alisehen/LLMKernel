import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B + bias  (row-major A[M, K], row-major C[M, N], B stored as [K, N])
    A: [M, K]
    B: [K, N]
    bias: [N]
    C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=out_mask)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    High-performance Linear: y = x @ weight.T + bias
    Works for arbitrary leading dims; contracts over last dim of x.

    x: [..., in_features]
    weight: [out_features, in_features]
    bias: [out_features]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Triton linear requires CUDA tensors"
    assert x.dtype == weight.dtype == bias.dtype == torch.float32, "Kernel currently supports float32"

    *prefix, in_features = x.shape
    out_features, in_features_w = weight.shape
    assert in_features == in_features_w, "Incompatible shapes for linear"

    x_2d = x.reshape(-1, in_features)  # [M, K]
    M, K = x_2d.shape
    N = out_features

    # B is [K, N]
    weight_t = weight.t().contiguous()

    y_2d = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_bias_kernel[grid](
        x_2d, weight_t, bias, y_2d,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        y_2d.stride(0), y_2d.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    y = y_2d.reshape(*prefix, N)
    return y


class ModelNew(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 channels=3, dropout=0.1, emb_dropout=0.1):
        """
        Vision Transformer (ViT) model with Triton-optimized Linear layers
        for patch embedding and MLP head.
        """
        super(ModelNew, self).__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # Keep nn.Linear for parameter initialization, but use Triton in forward.
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout),
            num_layers=depth
        )

        self.to_cls_token = nn.Identity()
        # Keep structure; we'll invoke Triton for the Linear layers in forward().
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img):
        """
        Forward pass of the Vision Transformer.

        :param img: Input image tensor, shape (batch_size, channels, image_size, image_size).
        :return: Output tensor, shape (batch_size, num_classes).
        """
        p = self.patch_size

        # Patchify: [B, C, H, W] -> [B, num_patches, patch_dim]
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.reshape(img.shape[0], -1, p * p * img.shape[1])

        # Triton-optimized patch embedding: Linear(patch_dim -> dim)
        x = triton_linear(x, self.patch_to_embedding.weight, self.patch_to_embedding.bias)

        # Add CLS token
        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding + dropout
        x = x + self.pos_embedding
        x = self.dropout(x)

        # Transformer encoder (PyTorch)
        x = self.transformer(x)

        # CLS token
        x = self.to_cls_token(x[:, 0])

        # MLP head with Triton-optimized Linear layers
        # First Linear
        x = triton_linear(x, self.mlp_head[0].weight, self.mlp_head[0].bias)
        # GELU + Dropout as in original
        x = self.mlp_head[1](x)
        x = self.mlp_head[2](x)
        # Final Linear
        x = triton_linear(x, self.mlp_head[3].weight, self.mlp_head[3].bias)

        return x
