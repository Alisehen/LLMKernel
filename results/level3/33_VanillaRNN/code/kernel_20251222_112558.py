# <complete ModelNew code with optimized Triton kernels>
import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def fused_i2h_tanh_kernel(
    x_ptr, h_ptr, w_ptr, b_ptr, out_ptr,
    B, I, H, O,
    stride_xm, stride_xk,
    stride_hm, stride_hk,
    stride_wo, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tile of output [B, O]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # batch indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # hidden/output indices
    offs_k = tl.arange(0, BLOCK_K)                    # reduction indices

    # Accumulator in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- First part of K: input x (length I) ----
    a_ptrs_x = x_ptr + (offs_m[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)
    b_ptrs_x = w_ptr + (offs_n[None, :] * stride_wo) + (offs_k[:, None] * stride_wk)

    for k in range(0, I, BLOCK_K):
        # Masks for valid loads
        a_mask = (offs_m[:, None] < B) & (offs_k[None, :] + k < I)
        b_mask = (offs_k[:, None] + k < I) & (offs_n[None, :] < O)

        a = tl.load(a_ptrs_x, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs_x, mask=b_mask, other=0.0)

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs_x += BLOCK_K * stride_xk
        b_ptrs_x += BLOCK_K * stride_wk

    # ---- Second part of K: hidden h (length H), W offset by I ----
    w_ptr_h = w_ptr + I * stride_wk
    a_ptrs_h = h_ptr + (offs_m[:, None] * stride_hm) + (offs_k[None, :] * stride_hk)
    b_ptrs_h = w_ptr_h + (offs_n[None, :] * stride_wo) + (offs_k[:, None] * stride_wk)

    for k in range(0, H, BLOCK_K):
        a_mask = (offs_m[:, None] < B) & (offs_k[None, :] + k < H)
        b_mask = (offs_k[:, None] + k < H) & (offs_n[None, :] < O)

        a = tl.load(a_ptrs_h, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs_h, mask=b_mask, other=0.0)

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs_h += BLOCK_K * stride_hk
        b_ptrs_h += BLOCK_K * stride_wk

    # Add bias: shape (O,)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < O, other=0.0)
    acc = acc + bias[None, :]

    # Tanh activation: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    two = 2.0
    e2x = tl.exp(acc * two)
    acc = (e2x - 1.0) / (e2x + 1.0)

    # Store result
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om) + (offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < B) & (offs_n[None, :] < O)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def linear_kernel(
    a_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wo, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ W^T + b
    # A: (M, K), W: (N, K) row-major, output C: (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
    b_ptrs = w_ptr + (offs_n[None, :] * stride_wo) + (offs_k[:, None] * stride_wk)

    for k in range(0, K, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        b_mask = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        a = a.to(tl.float32)
        b = b.to(tl.float32)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + (offs_m[:, None] * stride_om) + (offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def fused_i2h_tanh(x: torch.Tensor,
                   h: torch.Tensor,
                   weight: torch.Tensor,
                   bias: torch.Tensor) -> torch.Tensor:
    """
    Compute tanh((x, h) @ weight.T + bias) with Triton.
    x: (B, I)
    h: (B, H)
    weight: (hidden_size, I + H)
    bias: (hidden_size,)
    returns new_hidden: (B, hidden_size)
    """
    assert x.is_cuda and h.is_cuda and weight.is_cuda and bias.is_cuda
    B, I = x.shape
    B_h, H = h.shape
    assert B == B_h, "x and h batch size must match"

    O, K_tot = weight.shape
    assert K_tot == I + H, "weight second dim must equal input_size + hidden_size"
    assert bias.shape[0] == O

    x_c = x.contiguous()
    h_c = h.contiguous()
    w_c = weight.contiguous()

    out = torch.empty((B, O), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            max(1, triton.cdiv(B, meta['BLOCK_M'])),
            max(1, triton.cdiv(O, meta['BLOCK_N'])),
        )

    fused_i2h_tanh_kernel[grid](
        x_c, h_c, w_c, bias, out,
        B, I, H, O,
        x_c.stride(0), x_c.stride(1),
        h_c.stride(0), h_c.stride(1),
        w_c.stride(0), w_c.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=32, BLOCK_N=128, BLOCK_K=32,
        num_warps=8, num_stages=2,
    )
    return out


def linear_triton(a: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor) -> torch.Tensor:
    """
    Compute a @ weight.T + bias with Triton.
    a: (M, K)
    weight: (N, K)
    bias: (N,)
    returns: (M, N)
    """
    assert a.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = a.shape
    N, K_w = weight.shape
    assert K == K_w
    assert bias.shape[0] == N

    a_c = a.contiguous()
    w_c = weight.contiguous()
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (
            max(1, triton.cdiv(M, meta['BLOCK_M'])),
            max(1, triton.cdiv(N, meta['BLOCK_N'])),
        )

    linear_kernel[grid](
        a_c, w_c, bias, out,
        M, N, K,
        a_c.stride(0), a_c.stride(1),
        w_c.stride(0), w_c.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=32, BLOCK_N=128, BLOCK_K=32,
        num_warps=8, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Vanilla RNN with Triton-optimized kernels for the two Linear layers
        and fused input+hidden -> hidden + tanh.
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Keep nn.Linear for parameter management / initialization, but
        # do the compute in Triton.
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

        # Persistent hidden state buffer (initialized lazily)
        self.hidden = None

    def forward(self, x: torch.Tensor, initial_hidden: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch_size, input_size)
        initial_hidden: (batch_size, hidden_size) or None
        returns: (batch_size, output_size)
        """
        assert x.is_cuda, "ModelNew expects CUDA tensors for best performance"

        B = x.shape[0]

        if initial_hidden is not None:
            self.hidden = initial_hidden.to(device=x.device, dtype=x.dtype)
        elif self.hidden is None or self.hidden.shape[0] != B:
            # Initialize hidden to zeros if not provided or shape changed
            self.hidden = torch.zeros(B, self.hidden_size,
                                      device=x.device, dtype=x.dtype)
        else:
            self.hidden = self.hidden.to(device=x.device, dtype=x.dtype)

        # Fused: concat(x, h) -> Linear(i2h) -> tanh
        self.hidden = fused_i2h_tanh(
            x, self.hidden,
            self.i2h.weight, self.i2h.bias
        )

        # Hidden to output: Linear(h2o)
        output = linear_triton(
            self.hidden,
            self.h2o.weight, self.h2o.bias
        )

        return output
