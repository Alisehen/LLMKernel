import math
import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv2d_hswish_gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, H_in, W_in,
    C_out, H_out, W_out,
    subtract_value,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    C_IN: tl.constexpr, K_H: tl.constexpr, K_W: tl.constexpr,
):
    """
    GEMM-style implicit-im2col conv2d fused with:
      bias add, subtract_value, and HardSwish activation.

    Matmul layout:
      A: [M, K]  where M = N * H_out * W_out, K = C_IN * K_H * K_W (implicit im2col from x)
      B: [K, C_out] (pre-packed on host from original weight)
      Y: [M, C_out] -> reshape back to [N, C_out, H_out, W_out]
    """
    pid_m = tl.program_id(0)  # rows: M dimension (N * H_out * W_out)
    pid_n = tl.program_id(1)  # cols: C_out dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    HW_out = H_out * W_out
    P = N * HW_out  # total M

    mask_m = offs_m < P
    mask_n = offs_n < C_out

    # Decode flattened output index m -> (n, oh, ow)
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_TOTAL = C_IN * K_H * K_W  # reduction dimension

    # Loop over K in tiles of BLOCK_K
    for k0 in range(0, K_TOTAL, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]
        mask_k = offs_k < K_TOTAL

        # Decode K index -> (ic, kh, kw)
        kk = offs_k
        ic = kk // (K_H * K_W)
        rem_k = kk % (K_H * K_W)
        kh = rem_k // K_W
        kw = rem_k % K_W

        # Build broadcasted indices for input A: [BM, BK]
        n_b = n_idx[:, None]            # [BM,1]
        oh_b = oh_idx[:, None]          # [BM,1]
        ow_b = ow_idx[:, None]          # [BM,1]

        ic_b = ic[None, :]              # [1,BK]
        kh_b = kh[None, :]              # [1,BK]
        kw_b = kw[None, :]              # [1,BK]

        ih = oh_b + kh_b                # [BM,BK]
        iw = ow_b + kw_b                # [BM,BK]

        # Flattened input index: x[n, ic, ih, iw] in NCHW
        in_index = (((n_b * C_IN + ic_b) * H_in + ih) * W_in + iw)  # [BM,BK]

        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(x_ptr + in_index, mask=a_mask, other=0.0)
        a = a.to(tl.float32)  # [BM,BK]

        # B tile: w_ptr is [K_TOTAL, C_out] row-major
        b_index = offs_k[:, None] * C_out + offs_n[None, :]  # [BK,BN]
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(w_ptr + b_index, mask=b_mask, other=0.0)
        b = b.to(tl.float32)  # [BK,BN]

        # Matmul accumulate
        acc += tl.dot(a, b, allow_tf32=True)

    # Add bias
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)  # [BN]
    acc += bias_vals[None, :]

    # Subtract scalar
    acc = acc - subtract_value

    # HardSwish: x * clamp(x+3, 0, 6) / 6
    tmp = acc + 3.0
    tmp = tl.maximum(tmp, 0.0)
    tmp = tl.minimum(tmp, 6.0)
    acc = acc * (tmp * (1.0 / 6.0))

    # Store back to Y[N, C_out, H_out, W_out]
    n2 = n_idx[:, None]   # [BM,1]
    oh2 = oh_idx[:, None] # [BM,1]
    ow2 = ow_idx[:, None] # [BM,1]
    oc2 = offs_n[None, :] # [1,BN]

    out_index = (((n2 * C_out + oc2) * H_out + oh2) * W_out + ow2)  # [BM,BN]
    out_mask = mask_m[:, None] & mask_n[None, :]

    tl.store(y_ptr + out_index, acc, mask=out_mask)


def conv2d_hswish(x, weight, bias, subtract_value: float):
    """
    NCHW conv2d with kernel_size=K_H=K_W, stride=1, padding=0, dilation=1,
    fused with bias add, subtract_value, and HardSwish activation.

    This implementation uses a GEMM-style implicit-im2col convolution:
      M = N * H_out * W_out
      N = C_out
      K = C_in * K_H * K_W
    and performs Y = A @ B where A is implicitly gathered from x, and
    B is a pre-packed [K, C_out] matrix from weight.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in == C_in_w

    # For stride=1, padding=0
    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Pre-pack weight to [K, C_out] row-major for better matmul layout
    K_total = C_in * K_H * K_W
    # weight: [C_out, C_in, K_H, K_W] -> [C_out, K_total] -> [K_total, C_out]
    weight_2d = weight.view(C_out, K_total).transpose(0, 1).contiguous()

    P = N * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_hswish_gemm_kernel[grid](
        x, weight_2d, bias, y,
        N, H_in, W_in,
        C_out, H_out, W_out,
        float(subtract_value),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        C_IN=C_in, K_H=K_H, K_W=K_W,
        num_warps=4, num_stages=2,
    )

    return y


@triton.jit
def maxpool2d_mish_kernel(
    x_ptr, y_ptr,
    N, C, H_in, W_in,
    H_out, W_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    K_P: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)  # flattened (N * H_out * W_out)
    pid_n = tl.program_id(1)  # channels

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    P = N * H_out * W_out
    mask_m = offs_m < P
    mask_n = offs_n < C

    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    n = n_idx[:, None]   # [BM,1]
    oh = oh_idx[:, None] # [BM,1]
    ow = ow_idx[:, None] # [BM,1]
    c = offs_n[None, :]  # [1,BN]

    full_mask = mask_m[:, None] & mask_n[None, :]

    # initialize max accumulator
    acc = tl.full((BLOCK_M, BLOCK_N), -1e30, dtype=tl.float32)

    # MaxPool2d over K_P x K_P with stride=K_P, no padding
    for kh in range(0, K_P):
        ih = oh * K_P + kh
        for kw in range(0, K_P):
            iw = ow * K_P + kw
            in_index = (((n * C + c) * H_in + ih) * W_in + iw)  # [BM,BN]
            x_vals = tl.load(
                x_ptr + in_index,
                mask=full_mask,
                other=-1e30,
            )
            acc = tl.maximum(acc, x_vals)

    # Mish: x * tanh(softplus(x)), softplus(x) = log(1 + exp(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    t = (e2 - 1.0) / (e2 + 1.0)
    out_vals = acc * t

    out_index = (((n * C + c) * H_out + oh) * W_out + ow)  # [BM,BN]
    tl.store(y_ptr + out_index, out_vals, mask=full_mask)


def maxpool2d_mish(x, kernel_size: int):
    """
    NCHW MaxPool2d with kernel_size=stride=kernel_size, padding=0, dilation=1,
    fused with Mish activation.
    """
    assert x.is_cuda
    x = x.contiguous()

    N, C, H_in, W_in = x.shape
    K_P = int(kernel_size)

    # For stride=kernel_size, no padding:
    H_out = H_in // K_P
    W_out = W_in // K_P

    y = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    P = N * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C, meta["BLOCK_N"]),
        )

    maxpool2d_mish_kernel[grid](
        x, y,
        N, C, H_in, W_in,
        H_out, W_out,
        BLOCK_M=64, BLOCK_N=64,
        K_P=K_P,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized version of the target model:
    Conv2d -> subtract constant -> HardSwish -> MaxPool2d -> Mish
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        k = int(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = int(pool_kernel_size)

        # Parameters equivalent to nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, k, k)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Use a standard conv init for similar behavior
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Ensure tensors are on CUDA for Triton
        if not x.is_cuda:
            x = x.cuda()

        device = x.device
        if not self.weight.is_cuda:
            self.weight.data = self.weight.data.to(device)
        if not self.bias.is_cuda:
            self.bias.data = self.bias.data.to(device)

        x = conv2d_hswish(x, self.weight, self.bias, self.subtract_value)
        x = maxpool2d_mish(x, self.pool_kernel_size)
        return x
