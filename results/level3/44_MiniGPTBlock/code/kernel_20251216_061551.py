import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def new_gelu_kernel(
    x_ptr,        # *const T
    y_ptr,        # *mut T
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid over the flattened tensor
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input (masked for out-of-bounds)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute in fp32 for numerical stability
    x_f = x.to(tl.float32)

    # NewGELU: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    # Use a numerically stable tanh implementation:
    # tanh(v) = sign(v) * (1 - exp(-2*|v|)) / (1 + exp(-2*|v|))

    # x^3
    x2 = x_f * x_f
    x3 = x2 * x_f

    # u = x + 0.044715 * x^3
    coeff = 0.044715
    u = x_f + coeff * x3

    # v = sqrt(2/pi) * u
    sqrt_2_over_pi = 0.7978845608028654
    v = sqrt_2_over_pi * u

    # tanh(v) via stable exp formulation
    a = tl.abs(v)
    t = tl.exp(-2.0 * a)
    base = (1.0 - t) / (1.0 + t)
    sign = tl.where(v >= 0.0, 1.0, -1.0)
    tanh_v = sign * base

    y_f = 0.5 * x_f * (1.0 + tanh_v)

    # Cast back to original dtype and store
    y = y_f.to(x.dtype)
    tl.store(y_ptr + offsets, y, mask=mask)


def new_gelu_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of the NewGELU activation:
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    Operates on a flattened 1D view with a 1D grid for maximum coalescing.
    """
    assert x.is_cuda, "new_gelu_triton expects a CUDA tensor"
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)

    n_elements = x_contig.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    new_gelu_kernel[grid](
        x_contig,
        y,
        n_elements,
    )
    return y.view_as(x)


class NewGELU_Triton(nn.Module):
    """
    nn.Module wrapper around the Triton NewGELU kernel.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return new_gelu_triton(x)


class ModelNew(nn.Module):
    """Transformer block using Triton-accelerated NewGELU in the MLP."""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)

        # Keep structure similar to the original; only activation uses Triton
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU_Triton(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """
    Standard causal self-attention (kept in PyTorch for this version).
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y
