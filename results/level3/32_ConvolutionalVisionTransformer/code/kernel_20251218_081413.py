# <corrected code>
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math
import torch.nn.init as init


@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_stages=4,
            num_warps=8,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_kernel(
    a_ptr,  # [M, K]
    b_ptr,  # [K, N] (weight.T)
    bias_ptr,  # [N]
    c_ptr,  # [M, N]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B + bias, where:
      A: [M, K]
      B: [K, N]
      bias: [N]
      C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_remaining = K - k
        k_mask = offs_k < k_remaining

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        # Use full FP32 dot product to match PyTorch reference; disable TF32
        acc += tl.dot(a, b, allow_tf32=False)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Store result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of a fully-connected layer:
      y = x @ weight.T + bias

    x:      [B, in_features]
    weight: [out_features, in_features]
    bias:   [out_features]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA"

    B, K = x.shape
    out_features, K_w = weight.shape
    assert K == K_w, "Incompatible shapes for matrix multiplication"

    # Output tensor
    y = torch.empty((B, out_features), device=x.device, dtype=x.dtype)

    # B matrix is weight^T: [K, out_features]
    w_t = weight.t().contiguous()

    M = B
    N = out_features

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    fused_linear_kernel[grid](
        x,
        w_t,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_t.stride(0),
        w_t.stride(1),
        y.stride(0),
        y.stride(1),
    )

    return y


class ModelNew(nn.Module):
    """
    Convolutional Vision Transformer (CViT) with Triton-accelerated linear layers
    for the patch projection and final classification head.

    The linear layers are defined as nn.Linear modules so that their weights and
    biases can be tied or loaded identically to a reference PyTorch model. The
    Triton kernels operate directly on these shared parameters in forward().
    """

    def __init__(self, num_classes, embed_dim=512, num_heads=8,
                 num_layers=6, mlp_ratio=4.0, patch_size=4,
                 in_channels=3, image_size=32):
        super(ModelNew, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        # Patch embedding via conv
        self.conv1 = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2  # H/patch * W/patch

        # Linear projection: (embed_dim * num_patches) -> embed_dim
        # Defined as nn.Linear to stay in sync with reference model parameters.
        in_features_proj = embed_dim * num_patches
        out_features_proj = embed_dim
        self.linear_proj = nn.Linear(in_features_proj, out_features_proj, bias=True)

        # Transformer encoder layers (kept as PyTorch modules)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True,
            ) for _ in range(num_layers)
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Classification head: embed_dim -> num_classes
        # Also nn.Linear so weights can be tied or loaded from reference.
        self.fc_out = nn.Linear(embed_dim, num_classes, bias=True)

        # IMPORTANT: Do NOT reinitialize linear_proj / fc_out here.
        # This keeps ModelNew's parameter initialization identical to the
        # reference Model under the same RNG state, ensuring outputs match.

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, num_classes)
        """
        B = x.size(0)

        # Patch embedding
        x = self.conv1(x)           # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(start_dim=1)  # (B, embed_dim * num_patches)

        # Triton linear projection to embed_dim using shared nn.Linear params
        x = fused_linear(x, self.linear_proj.weight, self.linear_proj.bias)  # (B, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)        # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x.unsqueeze(1)), dim=1)   # (B, 2, embed_dim)

        # Transformer encoder stack
        for layer in self.transformer_layers:
            x = layer(x)  # (B, 2, embed_dim)

        # Classification head on CLS token, via Triton fused linear
        cls = x[:, 0]  # (B, embed_dim)
        out = fused_linear(cls, self.fc_out.weight, self.fc_out.bias)  # (B, num_classes)
        return out
