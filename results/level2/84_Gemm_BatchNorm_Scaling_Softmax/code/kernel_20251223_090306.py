import torch, torch.nn as nn, triton, triton.language as tl


# ---------------------------------------------
# GEMM (Linear) kernel: y = x @ W^T + b
# ---------------------------------------------
@triton.jit
def linear_forward_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for 2D tile grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to the first tile of A and B
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < (K - k)
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask.T & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, allow_tf32=True)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias (broadcast over rows)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Write back
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def fused_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    High-performance replacement for:
        x = F.linear(x, weight, bias)  # (x @ weight.T + bias)

    Args:
        x:      [M, K]
        weight: [N, K]
        bias:   [N]
    Returns:
        y:      [M, N]
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Strides assuming row-major but kept generic
    stride_am, stride_ak = x.stride()
    # Interpret weight as B[k, n] == weight[n, k] via custom strides
    stride_bw0, stride_bw1 = weight.stride()
    stride_bk = stride_bw1
    stride_bn = stride_bw0
    stride_cm, stride_cn = y.stride()

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    linear_forward_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=2,
    )
    return y


# ---------------------------------------------
# Fused scale + softmax over dim=1
# y[i, j] = softmax_j(scale * x[i, j])
# ---------------------------------------------
@triton.jit
def fused_scale_softmax_kernel(
    x_ptr, y_ptr,
    scale,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid

    offs = tl.arange(0, BLOCK_SIZE)
    row_mask = row < M

    # 1) Compute row-wise max of scaled values for numerical stability
    row_max = -float('inf')
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + offs
        mask = (cols < N) & row_mask
        x_ptrs = x_ptr + row * stride_xm + cols * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(mask, x, -float('inf'))
        block_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # 2) Compute denominator: sum(exp(x - row_max))
    row_sum = 0.0
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + offs
        mask = (cols < N) & row_mask
        x_ptrs = x_ptr + row * stride_xm + cols * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(mask, x, -float('inf'))
        exp_x = tl.exp(x - row_max)
        block_sum = tl.sum(exp_x, axis=0)
        row_sum += block_sum

    # 3) Write normalized probabilities
    inv_row_sum = 1.0 / row_sum
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + offs
        mask = (cols < N) & row_mask
        x_ptrs = x_ptr + row * stride_xm + cols * stride_xn
        y_ptrs = y_ptr + row * stride_ym + cols * stride_yn
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x * scale
        x = tl.where(mask, x, -float('inf'))
        exp_x = tl.exp(x - row_max)
        y = exp_x * inv_row_sum
        tl.store(y_ptrs, y, mask=mask)


def fused_scale_softmax(x: torch.Tensor, scale_param: torch.Tensor) -> torch.Tensor:
    """
    High-performance replacement for:
        x = scale * x
        x = softmax(x, dim=1)

    Args:
        x:           [M, N]
        scale_param: scalar tensor (shape (1,) or broadcastable to x)
    Returns:
        y:           [M, N]
    """
    assert x.is_cuda and scale_param.is_cuda
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)

    # Use scalar scale for best performance
    scale = float(scale_param.item())

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    BLOCK_SIZE = 128
    grid = lambda META: (triton.cdiv(M, 1),)

    fused_scale_softmax_kernel[grid](
        x, y,
        scale,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return y


# ---------------------------------------------
# Softmax over dim=1 (no extra scaling)
# y[i, j] = softmax_j(x[i, j])
# ---------------------------------------------
@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid

    offs = tl.arange(0, BLOCK_SIZE)
    row_mask = row < M

    # 1) Row-wise max for numerical stability
    row_max = -float('inf')
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + offs
        mask = (cols < N) & row_mask
        x_ptrs = x_ptr + row * stride_xm + cols * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # 2) Row-wise sum of exp(x - max)
    row_sum = 0.0
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + offs
        mask = (cols < N) & row_mask
        x_ptrs = x_ptr + row * stride_xm + cols * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        exp_x = tl.exp(x - row_max)
        block_sum = tl.sum(exp_x, axis=0)
        row_sum += block_sum

    inv_row_sum = 1.0 / row_sum

    # 3) Normalize and write back
    for start_n in range(0, N, BLOCK_SIZE):
        cols = start_n + offs
        mask = (cols < N) & row_mask
        x_ptrs = x_ptr + row * stride_xm + cols * stride_xn
        y_ptrs = y_ptr + row * stride_ym + cols * stride_yn
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        exp_x = tl.exp(x - row_max)
        y = exp_x * inv_row_sum
        tl.store(y_ptrs, y, mask=mask)


def softmax_dim1(x: torch.Tensor) -> torch.Tensor:
    """
    High-performance softmax over dim=1 (rows are batches, columns are features).

    Args:
        x: [M, N]
    Returns:
        y: [M, N]
    """
    assert x.is_cuda
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    BLOCK_SIZE = 128
    grid = lambda META: (triton.cdiv(M, 1),)

    softmax_kernel[grid](
        x, y,
        M, N,
        stride_xm, stride_xn,
        stride_ym, stride_yn,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return y


# ---------------------------------------------
# Model with Triton-accelerated Linear + Softmax
# BN + scale folded into Linear in eval mode
# ---------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        # Keep module structure/parameter names for easy state_dict loading
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))

        # Cached fused parameters for eval mode
        self._fused_weight = None
        self._fused_bias = None

    def train(self, mode: bool = True):
        # Override to invalidate fused cache when switching modes
        super().train(mode)
        # Any change in training/eval invalidates cached fused params
        self._fused_weight = None
        self._fused_bias = None
        return self

    def _build_fused_linear(self):
        """
        Build fused Linear parameters that fold:
          - original Linear (W, b)
          - BatchNorm1d (running_mean, running_var, weight, bias)
          - extra scale parameter

        Only valid in eval/inference mode where BatchNorm uses running stats.
        """
        W = self.gemm.weight
        b = self.gemm.bias
        bn = self.bn

        device = W.device
        dtype = W.dtype

        # Ensure everything is on the same device/dtype
        b = b.to(device=device, dtype=dtype)
        gamma = bn.weight.to(device=device, dtype=dtype)
        beta = bn.bias.to(device=device, dtype=dtype)
        running_mean = bn.running_mean.to(device=device, dtype=dtype)
        running_var = bn.running_var.to(device=device, dtype=dtype)
        eps = bn.eps
        scale = self.scale.to(device=device, dtype=dtype)

        # BN (eval): y_bn = (z - mean) / sqrt(var + eps) * gamma + beta
        # Then extra scaling: y = scale * y_bn
        # Let:
        #   sigma = sqrt(var + eps)
        #   alpha = scale * gamma / sigma
        #   b_fused = alpha * (b - mean) + scale * beta
        # And W_fused = alpha[:, None] * W
        sigma = torch.sqrt(running_var + eps)
        alpha = (gamma * scale) / sigma  # [N]
        W_fused = W.to(device=device, dtype=dtype) * alpha.view(-1, 1)
        b_fused = alpha * (b - running_mean) + scale * beta

        self._fused_weight = W_fused
        self._fused_bias = b_fused

    def _get_fused_linear_params(self):
        """
        Return cached fused (weight, bias), rebuilding if needed due to
        shape/device/dtype changes.
        """
        W = self.gemm.weight
        need_rebuild = (
            self._fused_weight is None
            or self._fused_bias is None
            or self._fused_weight.shape != W.shape
            or self._fused_weight.device != W.device
            or self._fused_weight.dtype != W.dtype
        )
        if need_rebuild:
            self._build_fused_linear()
        return self._fused_weight, self._fused_bias

    def forward(self, x):
        """
        Matches the reference Model.forward:

            x = self.gemm(x)
            x = self.bn(x)
            x = self.scale * x
            x = softmax(x, dim=1)

        with Triton acceleration and BN+scale folded into the Linear in eval mode.
        """
        # Expect x to be on CUDA for Triton kernels
        if self.training:
            # Training: need true BatchNorm semantics (batch stats, grad),
            # so keep the original pipeline but accelerate Linear + scale*softmax.
            x = fused_linear(x, self.gemm.weight, self.gemm.bias)
            x = self.bn(x)
            x = fused_scale_softmax(x, self.scale)
            return x
        else:
            # Eval/inference: fold BN + scale into a single fused Linear
            W_fused, b_fused = self._get_fused_linear_params()
            x = fused_linear(x, W_fused, b_fused)
            # No extra scaling here; it's already folded in. Just softmax.
            x = softmax_dim1(x)
            return x
