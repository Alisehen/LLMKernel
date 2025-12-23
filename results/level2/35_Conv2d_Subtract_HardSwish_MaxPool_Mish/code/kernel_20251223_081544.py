import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv2d_hswish_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    subtract_value,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    K_H: tl.constexpr, K_W: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)  # flattened (N * H_out * W_out)
    pid_n = tl.program_id(1)  # output channels

    # offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN]

    P = N * H_out * W_out
    mask_m = offs_m < P
    mask_n = offs_n < C_out

    # decode flat spatial index -> (n, oh, ow)
    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # convolution loops over input channel and kernel
    for ic in range(0, C_in):
        for kh in range(0, K_H):
            ih = oh_idx + kh  # [BM]
            for kw in range(0, K_W):
                iw = ow_idx + kw  # [BM]

                # input indices for this (ic, kh, kw)
                in_index = (((n_idx * C_in + ic) * H_in + ih) * W_in + iw)  # [BM]
                x_vals = tl.load(
                    x_ptr + in_index,
                    mask=mask_m,
                    other=0.0,
                )  # [BM]

                # weight indices: [C_out, C_in, K_H, K_W]
                w_index = (((offs_n * C_in + ic) * K_H + kh) * K_W + kw)  # [BN]
                w_vals = tl.load(
                    w_ptr + w_index,
                    mask=mask_n,
                    other=0.0,
                )  # [BN]

                # outer product accumulate
                acc += x_vals[:, None] * w_vals[None, :]

    # add bias
    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)  # [BN]
    acc += bias_vals[None, :]

    # subtract scalar
    acc = acc - subtract_value

    # HardSwish: x * clamp(x+3, 0, 6) / 6
    tmp = acc + 3.0
    tmp = tl.maximum(tmp, 0.0)
    tmp = tl.minimum(tmp, 6.0)
    acc = acc * (tmp * (1.0 / 6.0))

    # store output
    n2 = n_idx[:, None]           # [BM,1]
    oh2 = oh_idx[:, None]         # [BM,1]
    ow2 = ow_idx[:, None]         # [BM,1]
    oc2 = offs_n[None, :]         # [1,BN]

    out_index = (((n2 * C_out + oc2) * H_out + oh2) * W_out + ow2)  # [BM,BN]
    out_mask = mask_m[:, None] & mask_n[None, :]

    tl.store(y_ptr + out_index, acc, mask=out_mask)


def conv2d_hswish(x, weight, bias, subtract_value: float):
    """
    NCHW conv2d with kernel_size=K_H=K_W, stride=1, padding=0, dilation=1,
    fused with bias add, subtract_value, and HardSwish activation.
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in == C_in_w

    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    P = N * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_hswish_kernel[grid](
        x, weight, bias, y,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        float(subtract_value),
        BLOCK_M=64, BLOCK_N=64,
        K_H=K_H, K_W=K_W,
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
    # tanh(sp) = (exp(2*sp) - 1) / (exp(2*sp) + 1)
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
        # Expect x on CUDA for Triton
        if not x.is_cuda:
            x = x.cuda()

        x = conv2d_hswish(x, self.weight, self.bias, self.subtract_value)
        x = maxpool2d_mish(x, self.pool_kernel_size)
        return x
