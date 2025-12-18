import math
import torch
import torch.nn as nn
import torch.nn.init as init
import triton
import triton.language as tl


@triton.jit
def linear_gemm_bias_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tiling over (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for rows (M) and cols (N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for X (M x K) and W (K x N), both may be non-contiguous
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w, allow_tf32=True)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias: b has shape (N,)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Write back
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y = acc.to(tl.float32)  # output dtype cast happens on PyTorch side if needed
    tl.store(y_ptrs, y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:      (M, K)
    weight: (N, K)  (PyTorch Linear weight)
    bias:   (N,)
    returns (M, N)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "All tensors must be on CUDA device"

    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, "Incompatible shapes for x and weight"
    assert bias.shape[0] == N, "Bias shape must match out_features"

    # Prepare output
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # We compute x @ weight.T, so form W^T (K, N)
    # This is a one-time cheap transpose for this matmul.
    w_t = weight.t().contiguous()

    # Strides
    stride_xm, stride_xk = x.stride()
    stride_wk, stride_wn = w_t.stride()
    stride_ym, stride_yn = y.stride()

    # Launch config
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    linear_gemm_bias_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    # Match input dtype
    if x.dtype != torch.float32:
        y = y.to(x.dtype)
    return y


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with a Triton-accelerated final linear layer.
        """
        super(ModelNew, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        # Final linear layer parameters: (output_size, hidden_size)
        self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(output_size))

        # Initialize similar to nn.Linear defaults
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, x, h0=None, c0=None):
        """
        x:  (batch_size, sequence_length, input_size)
        h0: (num_layers, batch_size, hidden_size) or None
        c0: (num_layers, batch_size, hidden_size) or None
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        if h0 is None:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        if c0 is None:
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)

        out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden_size)
        last = out[:, -1, :]             # (batch, hidden_size), possibly non-contiguous

        # Triton linear: (batch, hidden_size) -> (batch, output_size)
        y = triton_linear(last, self.weight, self.bias)
        return y
