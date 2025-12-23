# <complete ModelNew code with optimized Triton kernels>

import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_i2h_tanh_kernel(
    x_ptr, h_ptr, w_ptr, bias_ptr, out_ptr,
    M, K1, K2, N,
    stride_xm, stride_xk,
    stride_hm, stride_hk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute: out = tanh( x @ W_x^T + h @ W_h^T + bias )
    where W = [W_x | W_h] along input-dimension.
    Shapes (PyTorch convention):
      x      : [M, K1]
      h      : [M, K2]
      W      : [N, K1 + K2]
      bias   : [N]
      out    : [M, N]
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -----------------------
    # Contribution from x @ W_x^T (first K1 columns of W)
    # -----------------------
    for k1 in range(0, K1, BLOCK_K):
        offs_k1 = k1 + tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k1[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k1[:, None] * stride_wk

        mask_x = (offs_m[:, None] < M) & (offs_k1[None, :] < K1)
        mask_w = (offs_k1[:, None] < K1) & (offs_n[None, :] < N)

        a = tl.load(x_ptrs, mask=mask_x, other=0.0)
        b = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    # -----------------------
    # Contribution from h @ W_h^T (last K2 columns of W)
    # -----------------------
    w_h_base = w_ptr + K1 * stride_wk

    for k2 in range(0, K2, BLOCK_K):
        offs_k2 = k2 + tl.arange(0, BLOCK_K)

        h_ptrs = h_ptr + offs_m[:, None] * stride_hm + offs_k2[None, :] * stride_hk
        w2_ptrs = w_h_base + offs_n[None, :] * stride_wn + offs_k2[:, None] * stride_wk

        mask_h = (offs_m[:, None] < M) & (offs_k2[None, :] < K2)
        mask_w2 = (offs_k2[:, None] < K2) & (offs_n[None, :] < N)

        a2 = tl.load(h_ptrs, mask=mask_h, other=0.0)
        b2 = tl.load(w2_ptrs, mask=mask_w2, other=0.0)

        acc += tl.dot(a2, b2, allow_tf32=True)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Tanh activation (manual implementation)
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    u = 2.0 * acc
    e = tl.exp(u)
    out_val = (e - 1.0) / (e + 1.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out_val, mask=mask_out)


@triton.jit
def fused_linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute: out = x @ W^T + bias
    Shapes (PyTorch convention):
      x    : [M, K]
      W    : [N, K]
      bias : [N]
      out  : [M, N]
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(x_ptrs, mask=mask_x, other=0.0)
        b = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


def fused_i2h_tanh(x: torch.Tensor,
                   h: torch.Tensor,
                   weight: torch.Tensor,
                   bias: torch.Tensor) -> torch.Tensor:
    """
    x      : [batch, input_size]
    h      : [batch, hidden_size]
    weight : [hidden_size, input_size + hidden_size]
    bias   : [hidden_size]
    """
    assert x.is_cuda and h.is_cuda and weight.is_cuda and bias.is_cuda
    M, K1 = x.shape
    Mh, K2 = h.shape
    assert Mh == M, "x and h must have same batch size"
    N, K_total = weight.shape
    assert K_total == K1 + K2, "weight second dim must equal input_size + hidden_size"
    assert bias.shape[0] == N

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    fused_i2h_tanh_kernel[grid](
        x, h, weight, bias, out,
        M, K1, K2, N,
        x.stride(0), x.stride(1),
        h.stride(0), h.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out


def fused_linear(x: torch.Tensor,
                 weight: torch.Tensor,
                 bias: torch.Tensor) -> torch.Tensor:
    """
    x      : [batch, in_features]
    weight : [out_features, in_features]
    bias   : [out_features]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w
    assert bias.shape[0] == N

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    fused_linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Triton-optimized Vanilla RNN cell:
          hidden_t = tanh( [x_t, h_{t-1}] @ W_ih^T + b_ih )
          y_t      = hidden_t @ W_ho^T + b_ho
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Hidden state will be lazily initialized based on the input batch size
        self.hidden = None

        # Parameters replacing nn.Linear layers
        self.i2h_weight = nn.Parameter(torch.empty(hidden_size, input_size + hidden_size))
        self.i2h_bias = nn.Parameter(torch.empty(hidden_size))
        self.h2o_weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.h2o_bias = nn.Parameter(torch.empty(output_size))

        # Initialize like nn.Linear
        for w, b in ((self.i2h_weight, self.i2h_bias),
                     (self.h2o_weight, self.h2o_bias)):
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(b, -bound, bound)

    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        """
        x: [batch_size, input_size]
        initial_hidden: [batch_size, hidden_size] or None
        Returns:
          output: [batch_size, output_size]
        """
        batch_size = x.shape[0]

        # Lazy initialization of hidden state, matching batch/device/dtype
        if (self.hidden is None or
                self.hidden.shape[0] != batch_size or
                self.hidden.shape[1] != self.hidden_size or
                self.hidden.device != x.device or
                self.hidden.dtype != x.dtype):
            # Use random initialization as in the original implementation
            self.hidden = torch.randn(
                batch_size, self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )

        # If an explicit initial hidden state is provided, use it
        if initial_hidden is not None:
            init_h = initial_hidden.to(device=x.device, dtype=x.dtype)
            if init_h.shape != self.hidden.shape:
                raise ValueError(
                    f"initial_hidden shape {init_h.shape} does not match "
                    f"expected {self.hidden.shape}"
                )
            self.hidden.copy_(init_h)

        # Ensure hidden is on the same device as x
        self.hidden = self.hidden.to(x.device)

        # Hidden state update via fused Triton kernel
        self.hidden = fused_i2h_tanh(
            x,
            self.hidden,
            self.i2h_weight,
            self.i2h_bias,
        )

        # Output projection via fused Triton linear kernel
        output = fused_linear(
            self.hidden,
            self.h2o_weight,
            self.h2o_bias,
        )
        return output
