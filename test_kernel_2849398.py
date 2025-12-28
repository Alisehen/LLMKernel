# <complete ModelNew code with optimized Triton kernels>
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# =========================
# Triton kernels
# =========================

@triton.jit
def patchify_kernel(
    x_ptr,          # *f32, input: (B, C, H, W)
    patches_ptr,    # *f32, output: (M, K) where M=B*PH*PW, K=C*PATCH*PATCH
    B, C, H, W,
    PH, PW, PATCH,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_ob, stride_ok,
    M, K,
    BLOCK_M: tl.constexpr,  # rows (patches)
    BLOCK_K: tl.constexpr,  # cols (flattened patch features)
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]

    m = offs_m[:, None]
    k = offs_k[None, :]

    patches_per_img = PH * PW
    patch_area = PATCH * PATCH

    # Decode patch row index -> (b, py, px)
    b = m // patches_per_img
    p_in_img = m - b * patches_per_img
    py = p_in_img // PW
    px = p_in_img - py * PW

    # Decode flattened patch column -> (c, ky, kx)
    c = k // patch_area
    kk = k - c * patch_area
    ky = kk // PATCH
    kx = kk - ky * PATCH

    h = py * PATCH + ky
    w = px * PATCH + kx

    x_ptrs = (
        x_ptr
        + b * stride_xb
        + c * stride_xc
        + h * stride_xh
        + w * stride_xw
    )
    out_ptrs = patches_ptr + m * stride_ob + k * stride_ok

    vals = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def linear_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # pointers for this block
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction along K
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k),
            other=0.0
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # add bias (broadcast over rows)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # write back
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# =========================
# Triton wrappers
# =========================

def triton_patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    x: (B, C, H, W)
    returns patches: (B * num_patches, C * patch_size * patch_size)
    """
    assert x.is_cuda, "Input must be on CUDA for Triton kernels."
    B, C, H, W = x.shape
    PH = H // patch_size
    PW = W // patch_size
    num_patches = PH * PW

    M = B * num_patches
    K = C * patch_size * patch_size

    patches = torch.empty((M, K), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(K, META["BLOCK_K"]),
    )

    patchify_kernel[grid](
        x, patches,
        B, C, H, W,
        PH, PW, patch_size,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        patches.stride(0), patches.stride(1),
        M, K,
        BLOCK_M=64,
        BLOCK_K=64,
    )

    return patches


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: (M, K)
    weight: (N, K)  (same layout as nn.Linear.weight)
    bias: (N,)
    returns: (M, N)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be CUDA tensors."

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"Incompatible shapes: x ({M}, {K}) vs weight ({N}, {K_w})"

    # Triton kernel expects B in shape (K, N)
    b = weight.t().contiguous()  # (K, N)
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_bias_kernel[grid](
        x, b, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )

    return y


# =========================
# Optimized Model
# =========================

class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6,
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        """
        Triton-optimized Convolutional Vision Transformer (CViT).
        Matches the high-level structure of the original Model, but
        replaces the patch embedding conv, linear projection, and final
        classifier with high-performance Triton kernels.
        """
        super(ModelNew, self).__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

        # ---- Conv1 equivalent parameters (Conv2d with kernel=stride=patch_size) ----
        # Weight: (embed_dim, in_channels, patch_size, patch_size)
        self.conv_weight = nn.Parameter(
            torch.empty(embed_dim, in_channels, patch_size, patch_size)
        )
        self.conv_bias = nn.Parameter(torch.empty(embed_dim))

        # Initialize similar to nn.Conv2d default (kaiming_uniform_)
        fan_in = in_channels * patch_size * patch_size
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv_bias, -bound, bound)

        # ---- Linear projection after flattening conv output ----
        # nn.Linear(embed_dim * num_patches, embed_dim)
        self.linear_proj_weight = nn.Parameter(
            torch.empty(embed_dim, embed_dim * num_patches)
        )
        self.linear_proj_bias = nn.Parameter(torch.empty(embed_dim))

        fan_in_lp = embed_dim * num_patches
        nn.init.kaiming_uniform_(self.linear_proj_weight, a=math.sqrt(5))
        bound_lp = 1 / math.sqrt(fan_in_lp)
        nn.init.uniform_(self.linear_proj_bias, -bound_lp, bound_lp)

        # ---- Transformer encoder layers (kept as PyTorch modules) ----
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # ---- CLS token ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---- Final classification head ----
        # nn.Linear(embed_dim, num_classes)
        self.fc_weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        self.fc_bias = nn.Parameter(torch.empty(num_classes))

        fan_in_fc = embed_dim
        nn.init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
        bound_fc = 1 / math.sqrt(fan_in_fc)
        nn.init.uniform_(self.fc_bias, -bound_fc, bound_fc)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, num_classes)
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, "Input channels mismatch."
        assert H == self.image_size and W == self.image_size, "Image size mismatch."

        # ---- Patch embedding via Triton patchify + linear ----
        # 1) Patchify input: (B, C, H, W) -> (B * num_patches, C * patch_size * patch_size)
        patches = triton_patchify(x, self.patch_size)

        # 2) Apply conv1 via GEMM: patches @ conv_weight_flat^T + conv_bias
        conv_w_flat = self.conv_weight.view(self.conv_weight.shape[0], -1)  # (embed_dim, C*P*P)
        conv_out_flat = triton_linear(patches, conv_w_flat, self.conv_bias)  # (B*num_patches, embed_dim)

        # 3) Reshape to (B, embed_dim, H/patch, W/patch)
        PH = H // self.patch_size
        PW = W // self.patch_size
        num_patches = PH * PW

        conv_out = conv_out_flat.view(B, num_patches, self.embed_dim)           # (B, P, E)
        conv_out = conv_out.view(B, PH, PW, self.embed_dim).permute(0, 3, 1, 2) # (B, E, PH, PW)

        # ---- Flatten spatial dimensions and project to embed_dim via Triton ----
        x_flat = conv_out.reshape(B, self.embed_dim * num_patches)  # (B, E * P)
        x_proj = triton_linear(x_flat, self.linear_proj_weight, self.linear_proj_bias)  # (B, E)

        # ---- Prepend CLS token and run transformer encoder layers ----
        cls_tokens = self.cls_token.expand(B, -1, -1)     # (B, 1, E)
        x_seq = torch.cat((cls_tokens, x_proj.unsqueeze(1)), dim=1)  # (B, 2, E)

        for layer in self.transformer_layers:
            x_seq = layer(x_seq)

        # ---- Classification head via Triton linear ----
        cls_repr = x_seq[:, 0]  # (B, E)
        logits = triton_linear(cls_repr, self.fc_weight, self.fc_bias)  # (B, num_classes)
        return logits
