import torch
import torch.nn as nn
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Implement missing Triton math helpers on top of tl.*
# -----------------------------------------------------------------------------


@triton.jit
def _tl_sigmoid(x):
    """
    Numerically-stable sigmoid implemented in Triton:
      sigmoid(x) = 1 / (1 + exp(-x))                if x >= 0
                 = exp(x) / (1 + exp(x))            if x < 0
    This avoids exp() on large positive values.
    """
    x_f32 = x.to(tl.float32)
    zero = 0.0
    is_pos = x_f32 >= zero

    exp_neg_x = tl.exp(-x_f32)   # for x >= 0 -> argument <= 0
    exp_x = tl.exp(x_f32)        # for x < 0  -> argument <= 0

    res_pos = 1.0 / (1.0 + exp_neg_x)
    res_neg = exp_x / (1.0 + exp_x)
    out = tl.where(is_pos, res_pos, res_neg)
    return out.to(x.dtype)


@triton.jit
def _tl_tanh(x):
    """
    Tanh via numerically-stable sigmoid:
      tanh(x) = 2 * sigmoid(2x) - 1
    """
    x_f32 = x.to(tl.float32)
    y = 2.0 * _tl_sigmoid(2.0 * x_f32) - 1.0
    return y.to(x.dtype)


@triton.jit
def _tl_gelu(x):
    """
    Approximate GELU:
      0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))
    Uses stable tanh to avoid NaNs.
    """
    x_f32 = x.to(tl.float32)
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x3 = x_f32 * x_f32 * x_f32
    inner = c0 * (x_f32 + 0.044715 * x3)
    tanh_inner = _tl_tanh(inner)
    out = 0.5 * x_f32 * (1.0 + tanh_inner)
    return out.to(x.dtype)


@triton.jit
def _tl_silu(x):
    """
    SiLU / Swish: x * sigmoid(x)
    """
    x_f32 = x.to(tl.float32)
    out = x_f32 * _tl_sigmoid(x_f32)
    return out.to(x.dtype)


@triton.jit
def _tl_softmax(x, axis: tl.constexpr = -1):
    """
    Simple, numerically-stable softmax over a given axis.
    This is meant for use inside Triton kernels where 'axis'
    is known at compile time.
    """
    x_f32 = x.to(tl.float32)

    # Support negative axis as "from the end"
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
    """
    Mish: x * tanh(softplus(x))
    softplus(x) is implemented in a reasonably stable way.
    """
    x_f32 = x.to(tl.float32)
    zero = 0.0
    is_pos = x_f32 > zero

    # For x > 0: softplus = x + log1p(exp(-x))  (argument of exp <= 0)
    exp_neg_x = tl.exp(-x_f32)
    softplus_pos = x_f32 + tl.log(1.0 + exp_neg_x)

    # For x <= 0: softplus = log1p(exp(x))      (x <= 0 -> argument of exp <= 0)
    exp_x = tl.exp(x_f32)
    softplus_neg = tl.log(1.0 + exp_x)

    softplus = tl.where(is_pos, softplus_pos, softplus_neg)
    out = x_f32 * _tl_tanh(softplus)
    return out.to(x.dtype)


# Expose them under tl.* names so other code can use tl.tanh / tl.gelu / ...
tl.sigmoid = _tl_sigmoid
tl.tanh = _tl_tanh
tl.gelu = _tl_gelu
tl.silu = _tl_silu
tl.softmax = _tl_softmax
tl.mish = _tl_mish


# -----------------------------------------------------------------------------
# High-performance Triton linear / linear+GELU kernels
# -----------------------------------------------------------------------------


@triton.jit
def _linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ADD_GELU: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_rem = K - k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N),
            other=0.0,
        )
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Optional GELU activation using numerically stable implementation
    if ADD_GELU:
        acc = _tl_gelu(acc)

    # Write back
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_out)


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    High-performance linear layer: y = x @ weight.T + bias
    x: [M, K], weight: [N, K], bias: [N]
    Output: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    # Weight is [N, K], need [K, N] for GEMM
    w_t = weight.t().contiguous()

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    _linear_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        ADD_GELU=False,
    )
    return y


def triton_linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    High-performance fused linear + GELU:
    y = GELU(x @ weight.T + bias)
    x: [M, K], weight: [N, K], bias: [N]
    Output: [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]

    w_t = weight.t().contiguous()
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    _linear_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        ADD_GELU=True,
    )
    return y


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

        # Patch embedding linear layer parameters (patch_dim -> dim)
        self.patch_weight = nn.Parameter(torch.randn(dim, patch_dim))
        self.patch_bias = nn.Parameter(torch.randn(dim))

        # Dropouts
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Transformer encoder (kept as PyTorch for structure / correctness)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
            ),
            num_layers=depth,
        )

        # MLP head parameters: dim -> mlp_dim -> num_classes
        self.mlp1_weight = nn.Parameter(torch.randn(mlp_dim, dim))
        self.mlp1_bias = nn.Parameter(torch.randn(mlp_dim))
        self.mlp2_weight = nn.Parameter(torch.randn(num_classes, mlp_dim))
        self.mlp2_bias = nn.Parameter(torch.randn(num_classes))
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [batch_size, channels, image_size, image_size]
        returns: [batch_size, num_classes]
        """
        b, c, h, w = img.shape
        p = self.patch_size

        # Extract patches: [b, c, h, w] -> [b, num_patches, patch_dim]
        x = img.unfold(2, p, p).unfold(3, p, p)
        # x: [b, c, h//p, w//p, p, p]
        x = x.contiguous().reshape(b, c, -1, p * p)
        x = x.permute(0, 2, 1, 3).reshape(b, -1, c * p * p)

        # Patch embedding via Triton linear
        x_flat = x.reshape(-1, x.shape[-1])
        x_emb = triton_linear(x_flat, self.patch_weight, self.patch_bias)
        x = x_emb.reshape(b, -1, x_emb.shape[-1])

        # Add class token and positional embeddings
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.emb_dropout(x)

        # Transformer encoder (expects [seq_len, batch, dim] by default)
        x = self.transformer(x)

        # Take CLS token
        x = x[:, 0]  # [b, dim]

        # MLP head with Triton-accelerated linears
        x_mlp_in = x.reshape(-1, x.shape[-1])
        x_hidden = triton_linear_gelu(x_mlp_in, self.mlp1_weight, self.mlp1_bias)
        x_hidden = self.mlp_dropout(x_hidden)
        x_out = triton_linear(x_hidden, self.mlp2_weight, self.mlp2_bias)
        x_out = x_out.reshape(b, -1)

        return x_out
