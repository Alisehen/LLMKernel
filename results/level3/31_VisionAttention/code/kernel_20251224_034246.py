import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ==============================
#  Triton kernels
# ==============================

@triton.jit
def linear_bias_triton_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Matrix multiplication with bias:
      Y[M, N] = X[M, K] @ W[K, N] + b[N]
    X: (M, K)
    W: (K, N)
    b: (N,)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first K-tile
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_block, w_block, allow_tf32=True)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]

    # Store result
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def fused_qkv_sdpa_triton_kernel(
    x_ptr,                 # (LB, E)
    w_q_ptr, w_k_ptr, w_v_ptr,
    b_q_ptr, b_k_ptr, b_v_ptr,
    o_ptr,                 # (BH, L, Dh)
    L, B, E, head_dim, num_heads,
    stride_xm, stride_xk,
    stride_wq0, stride_wq1,
    stride_wk0, stride_wk1,
    stride_wv0, stride_wv1,
    stride_obh, stride_om, stride_od,
    sm_scale,
    BLOCK_M: tl.constexpr,  # query tile size (along sequence)
    BLOCK_N: tl.constexpr,  # key tile size (along sequence)
    BLOCK_D: tl.constexpr,  # head dim tile (>= head_dim)
    BLOCK_K: tl.constexpr,  # embed dim tile
):
    """
    Fused QKV projection + scaled dot-product attention (FlashAttention-style):

    Input:
      x_ptr:   [L*B, E]  (row-major)
      in-proj weights/bias sliced as:
        w_q_ptr, w_k_ptr, w_v_ptr: each [E, E] (out_features, in_features)
        b_q_ptr, b_k_ptr, b_v_ptr: each [E]
      We interpret them internally to form, per-head:
        Q(l, b, h, :) = X(l, b, :) @ W_q^T[:, h*Dh:(h+1)*Dh] + b_q[h*Dh:(h+1)*Dh]
        K, V analogously.

    Output:
      o_ptr: [BH, L, Dh] contiguous
    """
    pid_m = tl.program_id(0)   # tile of queries (sequence index)
    pid_bh = tl.program_id(1)  # batch*head index

    # Derive batch and head indices
    b_idx = pid_bh // num_heads
    h_idx = pid_bh % num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # query sequence positions [0, L)
    offs_d = tl.arange(0, BLOCK_D)                    # head feature indices [0, head_dim)
    mask_m = offs_m < L
    mask_d = offs_d < head_dim

    head_offset = h_idx * head_dim  # starting channel offset within embed_dim

    # -----------------------
    # 1) Compute Q tile: (BLOCK_M, BLOCK_D)
    # -----------------------
    q = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for k0 in range(0, E, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < E

        # Load X queries: X_q[BLOCK_M, BLOCK_K]
        # row index in x_lin: m_idx = l * B + b_idx
        m_idx = offs_m * B + b_idx
        xq_ptrs = x_ptr + m_idx[:, None] * stride_xm + offs_k[None, :] * stride_xk
        xq = tl.load(
            xq_ptrs,
            mask=(mask_m[:, None] & k_mask[None, :]),
            other=0.0,
        )

        # Load W_q_in submatrix (K x Dh): weight_in[k, d] = W_q^T[k, head_offset + d]
        # Original layout W_q[out, in] -> row_stride=stride_wq0, col_stride=stride_wq1
        wq_ptrs = (
            w_q_ptr
            + (head_offset + offs_d[None, :]) * stride_wq0
            + offs_k[:, None] * stride_wq1
        )
        wq = tl.load(
            wq_ptrs,
            mask=(k_mask[:, None] & mask_d[None, :]),
            other=0.0,
        )

        q += tl.dot(xq, wq, allow_tf32=True)

    # Add Q bias
    bq = tl.load(b_q_ptr + head_offset + offs_d, mask=mask_d, other=0.0)
    q += bq[None, :]

    # -----------------------
    # 2) Flash-attention over keys/values
    # -----------------------
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, L, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < L

        # We'll compute:
        # K_T: (BLOCK_D, BLOCK_N)  = W_k_head^T(D,K) @ X_keys^T(K,N)
        # V:   (BLOCK_N, BLOCK_D)  = X_keys(N,K)     @ W_v_in(K,D)
        k_T = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
        v = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

        # row indices in x_lin for keys: n_idx = l * B + b_idx
        n_idx = offs_n * B + b_idx

        for k0 in range(0, E, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs_k < E

            # X_keys^T: (BLOCK_K, BLOCK_N)
            xk_T_ptrs = (
                x_ptr
                + n_idx[None, :] * stride_xm
                + offs_k[:, None] * stride_xk
            )
            xk_T = tl.load(
                xk_T_ptrs,
                mask=(k_mask[:, None] & mask_n[None, :]),
                other=0.0,
            )

            # W_k_head^T: (BLOCK_D, BLOCK_K) from original W_k[out, in]
            wk_T_ptrs = (
                w_k_ptr
                + (head_offset + offs_d[:, None]) * stride_wk0
                + offs_k[None, :] * stride_wk1
            )
            wk_T = tl.load(
                wk_T_ptrs,
                mask=(mask_d[:, None] & k_mask[None, :]),
                other=0.0,
            )

            # Accumulate K_T(D,N)
            k_T += tl.dot(wk_T, xk_T, allow_tf32=True)

            # X_keys: (BLOCK_N, BLOCK_K)
            xk_ptrs = (
                x_ptr
                + n_idx[:, None] * stride_xm
                + offs_k[None, :] * stride_xk
            )
            xk = tl.load(
                xk_ptrs,
                mask=(mask_n[:, None] & k_mask[None, :]),
                other=0.0,
            )

            # W_v_in: (BLOCK_K, BLOCK_D)
            wv_ptrs = (
                w_v_ptr
                + (head_offset + offs_d[None, :]) * stride_wv0
                + offs_k[:, None] * stride_wv1
            )
            wv = tl.load(
                wv_ptrs,
                mask=(k_mask[:, None] & mask_d[None, :]),
                other=0.0,
            )

            # Accumulate V(N,D)
            v += tl.dot(xk, wv, allow_tf32=True)

        # Add K/V bias
        bk = tl.load(b_k_ptr + head_offset + offs_d, mask=mask_d, other=0.0)
        k_T += bk[:, None]

        bv = tl.load(b_v_ptr + head_offset + offs_d, mask=mask_d, other=0.0)
        v += bv[None, :]

        # Scores: (BLOCK_M, BLOCK_N) = (BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_N)
        scores = tl.dot(q, k_T, allow_tf32=True) * sm_scale

        # Mask invalid sequence positions
        valid = mask_m[:, None] & mask_n[None, :]
        scores = tl.where(valid, scores, -1e9)

        # Online softmax
        block_max = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, block_max)
        p = tl.exp(scores - m_i_new[:, None])

        l_i_new = l_i * tl.exp(m_i - m_i_new) + tl.sum(p, axis=1)

        alpha = l_i * tl.exp(m_i - m_i_new) / l_i_new
        beta = 1.0 / l_i_new

        # (BLOCK_M, BLOCK_D) = (BLOCK_M, BLOCK_N) @ (BLOCK_N, BLOCK_D)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=True) * beta[:, None]

        m_i = m_i_new
        l_i = l_i_new

    # -----------------------
    # 3) Store output
    # -----------------------
    o_ptrs = (
        o_ptr
        + pid_bh * stride_obh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    o_mask = mask_m[:, None] & mask_d[None, :]
    tl.store(o_ptrs, acc, mask=o_mask)


@triton.jit
def layernorm_triton_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    M, N, eps,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    """
    LayerNorm over last dimension N:
      y[m, n] = (x[m, n] - mean_m) / sqrt(var_m + eps) * gamma[n] + beta[n]
    Assumes N <= BLOCK_N.
    """
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)

    mask = offs_n < N

    # Load row x[m, :]
    x_ptrs = x_ptr + pid * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N

    # Compute variance
    diff = x - mean
    diff = tl.where(mask, diff, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    y = diff * rstd

    # Affine transform
    gamma = tl.load(gamma_ptr + offs_n, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs_n, mask=mask, other=0.0)
    y = y * gamma + beta

    # Store result
    y_ptrs = y_ptr + pid * stride_ym + offs_n * stride_yn
    tl.store(y_ptrs, y, mask=mask)


# ==============================
#  Python wrappers
# ==============================

def triton_linear_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      (M, K)
    weight: (N, K)  (PyTorch Linear layout: out_features, in_features)
    bias:   (N,)
    returns (M, N)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Convert to (K, N) for GEMM: X[M,K] @ W[K,N]
    w_t = weight.t().contiguous()

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) if M > 0 else 1,
            triton.cdiv(N, meta["BLOCK_N"]) if N > 0 else 1,
        )

    linear_bias_triton_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w_t.stride(0), w_t.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
    )
    return y


def triton_fused_qkv_sdpa(
    x_lin: torch.Tensor,          # (L*B, E)
    in_proj_weight: torch.Tensor, # (3E, E)
    in_proj_bias: torch.Tensor,   # (3E,)
    B: int,
    num_heads: int,
) -> torch.Tensor:
    """
    Fused QKV projection + SDPA.

    x_lin:         (L*B, E), contiguous
    in_proj_weight: (3E, E), as in nn.MultiheadAttention.in_proj_weight
    in_proj_bias:   (3E,)
    returns:       (B*num_heads, L, Dh), contiguous
    """
    assert x_lin.is_cuda and in_proj_weight.is_cuda and in_proj_bias.is_cuda
    LB, E = x_lin.shape
    L = LB // B
    assert LB == L * B, "x_lin first dim must equal L*B"
    assert in_proj_weight.shape == (3 * E, E)
    assert in_proj_bias.shape[0] == 3 * E
    assert E % num_heads == 0
    Dh = E // num_heads

    # Slice Q, K, V weights and biases (no copies, just views)
    w_q = in_proj_weight[0:E, :]       # (E, E)
    w_k = in_proj_weight[E:2 * E, :]   # (E, E)
    w_v = in_proj_weight[2 * E:3 * E, :]  # (E, E)

    b_q = in_proj_bias[0:E]
    b_k = in_proj_bias[E:2 * E]
    b_v = in_proj_bias[2 * E:3 * E]

    BH = B * num_heads
    o = torch.empty((BH, L, Dh), device=x_lin.device, dtype=x_lin.dtype)

    sm_scale = 1.0 / math.sqrt(Dh)

    def grid(meta):
        return (
            triton.cdiv(L, meta["BLOCK_M"]) if L > 0 else 1,
            BH if BH > 0 else 1,
        )

    fused_qkv_sdpa_triton_kernel[grid](
        x_lin,
        w_q, w_k, w_v,
        b_q, b_k, b_v,
        o,
        L, B, E, Dh, num_heads,
        x_lin.stride(0), x_lin.stride(1),
        w_q.stride(0), w_q.stride(1),
        w_k.stride(0), w_k.stride(1),
        w_v.stride(0), w_v.stride(1),
        o.stride(0), o.stride(1), o.stride(2),
        sm_scale,
        BLOCK_M=64, BLOCK_N=64, BLOCK_D=64, BLOCK_K=64,
    )

    return o


def triton_layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """
    x:     (M, N)
    gamma: (N,)
    beta:  (N,)
    returns (M, N)
    """
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda
    M, N = x.shape
    y = torch.empty_like(x)

    def grid(meta):
        return (M if M > 0 else 1,)

    layernorm_triton_kernel[grid](
        x, gamma, beta, y,
        M, N, eps,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_N=1024,  # supports embed_dim up to 1024
    )
    return y


# ==============================
#  High-performance Model
# ==============================

class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Triton-optimized Attention Block with Multihead Self-Attention + LayerNorm.
        """
        super(ModelNew, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # In-projection (Q, K, V) combined: (3*E, E)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))

        # Out projection: (E, E)
        self.out_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.out_proj_bias = nn.Parameter(torch.zeros(embed_dim))

        # LayerNorm parameters
        self.ln_weight = nn.Parameter(torch.ones(embed_dim))
        self.ln_bias = nn.Parameter(torch.zeros(embed_dim))
        self.ln_eps = 1e-5

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.zeros_(self.out_proj_bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        E = self.embed_dim
        assert C == E, "Input channels must equal embed_dim"
        L = H * W

        # (B, C, H, W) -> (L, B, E)
        x_seq = x.view(B, C, L).permute(2, 0, 1).contiguous()  # (L, B, E)
        L_seq, B_seq, E_seq = x_seq.shape
        assert B_seq == B and E_seq == E

        # Prepare for linear: (L*B, E)
        x_lin = x_seq.view(L_seq * B_seq, E_seq).contiguous()

        # Fused QKV projection + SDPA via Triton
        context = triton_fused_qkv_sdpa(
            x_lin,
            self.in_proj_weight,
            self.in_proj_bias,
            B_seq,
            self.num_heads,
        )  # (BH, L, Dh)

        # Merge heads back: (BH, L, Dh) -> (L, B, E)
        Hh = self.num_heads
        Dh = self.head_dim
        BH = B_seq * Hh

        context = context.view(B_seq, Hh, L_seq, Dh).permute(2, 0, 1, 3).contiguous()
        context = context.view(L_seq, B_seq, E_seq)  # (L, B, E)

        # Output projection (unfused)
        context_lin = context.view(L_seq * B_seq, E_seq).contiguous()
        attn_output = triton_linear_bias(
            context_lin, self.out_proj_weight, self.out_proj_bias
        )
        attn_output = attn_output.view(L_seq, B_seq, E_seq)

        # Residual + LayerNorm
        residual = x_seq  # (L, B, E)
        y = attn_output + residual

        y_flat = y.view(L_seq * B_seq, E_seq).contiguous()
        # ln_weight/ln_bias are already Parameters; ensure they're on CUDA
        y_norm = triton_layernorm(
            y_flat, self.ln_weight.to(y_flat.device), self.ln_bias.to(y_flat.device), self.ln_eps
        )
        y_norm = y_norm.view(L_seq, B_seq, E_seq)

        # Back to (B, C, H, W)
        y_out = y_norm.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return y_out
