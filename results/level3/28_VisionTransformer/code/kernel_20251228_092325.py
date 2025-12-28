import torch, torch.nn as nn, triton, triton.language as tl


# -----------------------------------------------------------------------------
# Triton math helpers (manual implementations of missing tl.* functions)
# -----------------------------------------------------------------------------


@triton.jit
def _tl_sigmoid(x):
    x_f32 = x.to(tl.float32)
    zero = 0.0
    is_pos = x_f32 >= zero

    exp_neg_x = tl.exp(-x_f32)
    exp_x = tl.exp(x_f32)

    res_pos = 1.0 / (1.0 + exp_neg_x)
    res_neg = exp_x / (1.0 + exp_x)
    out = tl.where(is_pos, res_pos, res_neg)
    return out.to(x.dtype)


@triton.jit
def _tl_tanh(x):
    x_f32 = x.to(tl.float32)
    y = 2.0 * _tl_sigmoid(2.0 * x_f32) - 1.0
    return y.to(x.dtype)


@triton.jit
def _tl_gelu(x):
    x_f32 = x.to(tl.float32)
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x2 = x_f32 * x_f32
    x3 = x2 * x_f32
    inner = c0 * (x_f32 + 0.044715 * x3)
    tanh_inner = _tl_tanh(inner)
    out = 0.5 * x_f32 * (1.0 + tanh_inner)
    return out.to(x.dtype)


@triton.jit
def _tl_silu(x):
    x_f32 = x.to(tl.float32)
    out = x_f32 * _tl_sigmoid(x_f32)
    return out.to(x.dtype)


@triton.jit
def _tl_softmax(x, axis: tl.constexpr = -1):
    x_f32 = x.to(tl.float32)
    ndim = x_f32.ndim
    ax = tl.where(axis < 0, axis + ndim, axis)

    x_max = tl.max(x_f32, axis=ax)
    x_max = tl.broadcast_to(x_max, x_f32.shape)
    x_shifted = x_f32 - x_max
    exp_x = tl.exp(x_shifted)
    exp_sum = tl.sum(exp_x, axis=ax)
    exp_sum = tl.broadcast_to(exp_sum, x_f32.shape)
    out = exp_x / exp_sum
    return out.to(x.dtype)


@triton.jit
def _tl_mish(x):
    x_f32 = x.to(tl.float32)
    zero = 0.0
    is_pos = x_f32 > zero

    exp_neg_x = tl.exp(-x_f32)
    softplus_pos = x_f32 + tl.log(1.0 + exp_neg_x)

    exp_x = tl.exp(x_f32)
    softplus_neg = tl.log(1.0 + exp_x)

    softplus = tl.where(is_pos, softplus_pos, softplus_neg)
    out = x_f32 * _tl_tanh(softplus)
    return out.to(x.dtype)


# Expose as tl.* for convenience / compatibility
tl.sigmoid = _tl_sigmoid
tl.tanh = _tl_tanh
tl.gelu = _tl_gelu
tl.silu = _tl_silu
tl.softmax = _tl_softmax
tl.mish = _tl_mish


# -----------------------------------------------------------------------------
# High-performance Triton linear / linear+GELU kernels
#   - 2D grid over (M, N)
#   - Fused bias add + GELU share same offsets & mask as output store
#   - Tuned for RTX 4090 (Ada) with aggressive autotuning
# -----------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
            },
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
            },
            num_warps=8,
            num_stages=5,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ADD_GELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # -------------------------------------------------------------------------
    # 2D program id -> output tile (BLOCK_M x BLOCK_N)
    # -------------------------------------------------------------------------
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for output tile; ALL fused ops (bias, activation, store) use these
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_c = mask_m[:, None] & mask_n[None, :]

    # -------------------------------------------------------------------------
    # Pointers for A and B tiles
    #   A: [M, K] with strides (stride_am, stride_ak)
    #   B: [K, N] with strides (stride_bk, stride_bn)
    # -------------------------------------------------------------------------
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        # Remaining K elements
        k_rem = K - k
        k_mask = offs_k < k_rem

        # Load A and B with proper masks
        a = tl.load(
            a_ptrs,
            mask=(mask_m[:, None] & k_mask[None, :]),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_mask[:, None] & mask_n[None, :]),
            other=0.0,
        )

        # Accumulate
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b, allow_tf32=True)

        # Advance pointers along K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # -------------------------------------------------------------------------
    # Fused bias add + optional GELU
    #   All operations use the SAME output offsets and mask_c
    # -------------------------------------------------------------------------
    # Load bias for this N tile: shape [BLOCK_N]
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]

    if ADD_GELU:
        # GELU in fp32 for better numerical behavior
        acc = _tl_gelu(acc)

    # -------------------------------------------------------------------------
    # Store result: offsets & mask shared with fused ops above
    # -------------------------------------------------------------------------
    c_offsets = (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptr + c_offsets, acc, mask=mask_c)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    y = x @ weight.T + bias
    x: [M, K], weight: [N, K], bias: [N] -> y: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    # We want B as [K, N] contiguous for coalesced loads
    w_t = weight.t().contiguous()
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        return (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

    _linear_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        ADD_GELU=False,
    )
    return y


def triton_linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    y = GELU(x @ weight.T + bias)
    x: [M, K], weight: [N, K], bias: [N] -> y: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    w_t = weight.t().contiguous()
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        return (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

    _linear_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        ADD_GELU=True,
    )
    return y


# -----------------------------------------------------------------------------
# Custom Transformer encoder block with SDPA + Triton linears
# -----------------------------------------------------------------------------


class TritonViTEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by number of heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        # QKV projection fused: [3*dim, dim]
        self.qkv_weight = nn.Parameter(torch.randn(3 * dim, dim))
        self.qkv_bias = nn.Parameter(torch.randn(3 * dim))

        # Output projection: [dim, dim]
        self.out_proj_weight = nn.Parameter(torch.randn(dim, dim))
        self.out_proj_bias = nn.Parameter(torch.randn(dim))

        # MLP projections: dim -> mlp_dim -> dim
        self.mlp1_weight = nn.Parameter(torch.randn(mlp_dim, dim))
        self.mlp1_bias = nn.Parameter(torch.randn(mlp_dim))
        self.mlp2_weight = nn.Parameter(torch.randn(dim, mlp_dim))
        self.mlp2_bias = nn.Parameter(torch.randn(dim))

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)      # after attention
        self.mlp_dropout = nn.Dropout(dropout)   # between MLP layers
        self.dropout2 = nn.Dropout(dropout)      # after MLP

        # LayerNorms (post-norm, matching nn.TransformerEncoderLayer default)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq, dim]
        """
        b, s, d = x.shape
        # ---- Self-attention ----
        x_flat = x.reshape(b * s, d)
        qkv = triton_linear(x_flat, self.qkv_weight, self.qkv_bias)  # [b*s, 3*dim]
        qkv = qkv.reshape(b, s, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, b, h, s, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]  # [b, h, s, hd]

        # FlashAttention-style fused attention via SDPA (no dropout inside for speed)
        attn = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )  # [b, h, s, hd]

        attn = attn.permute(0, 2, 1, 3).reshape(b * s, d)  # [b*s, dim]
        attn_out = triton_linear(attn, self.out_proj_weight, self.out_proj_bias)
        attn_out = attn_out.reshape(b, s, d)
        attn_out = self.dropout1(attn_out)
        x = self.norm1(x + attn_out)

        # ---- MLP ----
        x_flat = x.reshape(b * s, d)
        hidden = triton_linear_gelu(x_flat, self.mlp1_weight, self.mlp1_bias)
        hidden = self.mlp_dropout(hidden)
        mlp_out = triton_linear(hidden, self.mlp2_weight, self.mlp2_bias)
        mlp_out = mlp_out.reshape(b, s, d)
        mlp_out = self.dropout2(mlp_out)
        x = self.norm2(x + mlp_out)

        return x


# -----------------------------------------------------------------------------
# Model definition
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
        super(ModelNew, self).__init__()

        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_size = patch_size

        # Positional and class token embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Patch embedding linear params (patch_dim -> dim)
        self.patch_weight = nn.Parameter(torch.randn(dim, patch_dim))
        self.patch_bias = nn.Parameter(torch.randn(dim))

        # Embedding dropout
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Stack of optimized Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TritonViTEncoderLayer(dim=dim, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
                for _ in range(depth)
            ]
        )

        # Classification head: dim -> mlp_dim -> num_classes
        self.mlp1_weight = nn.Parameter(torch.randn(mlp_dim, dim))
        self.mlp1_bias = nn.Parameter(torch.randn(mlp_dim))
        self.mlp2_weight = nn.Parameter(torch.randn(num_classes, mlp_dim))
        self.mlp2_bias = nn.Parameter(torch.randn(num_classes))
        self.head_dropout = nn.Dropout(dropout)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [batch_size, channels, image_size, image_size]
        return: [batch_size, num_classes]
        """
        b, c, h, w = img.shape
        p = self.patch_size

        # Extract patches: [b, c, h, w] -> [b, num_patches, patch_dim]
        x = img.unfold(2, p, p).unfold(3, p, p)
        # x: [b, c, h//p, w//p, p, p]
        x = x.contiguous().reshape(b, c, -1, p * p)
        x = x.permute(0, 2, 1, 3).reshape(b, -1, c * p * p)  # [b, num_patches, patch_dim]

        # Patch embedding via Triton linear
        x_flat = x.reshape(-1, x.shape[-1])  # [b*num_patches, patch_dim]
        x_emb = triton_linear(x_flat, self.patch_weight, self.patch_bias)
        x = x_emb.reshape(b, -1, x_emb.shape[-1])  # [b, num_patches, dim]

        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(b, -1, -1)  # [b, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b, num_patches+1, dim]
        x = x + self.pos_embedding
        x = self.emb_dropout(x)

        # Transformer encoder stack (batch, seq, dim)
        for layer in self.layers:
            x = layer(x)

        # CLS token
        x_cls = x[:, 0]  # [b, dim]

        # MLP head with Triton-accelerated linears
        x_mlp_in = x_cls.reshape(-1, x_cls.shape[-1])  # [b, dim]
        x_hidden = triton_linear_gelu(x_mlp_in, self.mlp1_weight, self.mlp1_bias)
        x_hidden = self.head_dropout(x_hidden)
        x_out = triton_linear(x_hidden, self.mlp2_weight, self.mlp2_bias)
        x_out = x_out.reshape(b, -1)  # [b, num_classes]

        return x_out
