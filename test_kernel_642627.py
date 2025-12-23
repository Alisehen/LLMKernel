import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv3x3_relu_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, H, W, C_out,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_y_n, stride_y_c, stride_y_h, stride_y_w,
    stride_w_oc, stride_w_ic, stride_w_ky, stride_w_kx,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    # Decode (n, oc) from flattened program id
    oc = pid_nc % C_out
    n = pid_nc // C_out

    # HW tiling
    hw_start = pid_hw * BLOCK_HW
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    HW = H * W
    mask_hw = offs_hw < HW

    h_idx = offs_hw // W
    w_idx = offs_hw % W

    # Accumulator
    acc = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    # Base output pointer offset for this (n, oc)
    y_base = n * stride_y_n + oc * stride_y_c

    # Loop over input channels
    for ic in range(0, C_in):
        x_base = n * stride_x_n + ic * stride_x_c
        w_base = oc * stride_w_oc + ic * stride_w_ic

        # 3x3 kernel
        for ky in range(0, 3):
            iy = h_idx + (ky - 1)
            mask_y = (iy >= 0) & (iy < H)

            for kx in range(0, 3):
                ix = w_idx + (kx - 1)
                mask_x = (ix >= 0) & (ix < W)

                m = mask_hw & mask_y & mask_x

                x_offs = x_base + iy * stride_x_h + ix * stride_x_w
                x_vals = tl.load(x_ptr + x_offs, mask=m, other=0.0)

                w_off = w_base + ky * stride_w_ky + kx * stride_w_kx
                w_val = tl.load(w_ptr + w_off)

                acc += x_vals * w_val

    # Add bias
    b_val = tl.load(b_ptr + oc)
    acc += b_val

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store
    y_offs = y_base + h_idx * stride_y_h + w_idx * stride_y_w
    tl.store(y_ptr + y_offs, acc, mask=mask_hw)


def conv3x3_relu_triton(x, weight, bias):
    """
    x:      (N, C_in, H, W)
    weight: (C_out, C_in, 3, 3)
    bias:   (C_out,)
    """
    assert x.ndim == 4
    assert weight.ndim == 4 and weight.shape[2] == 3 and weight.shape[3] == 3
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]

    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)

    BLOCK_HW = 128

    grid = lambda META: (
        N * C_out,
        triton.cdiv(H * W, META["BLOCK_HW"]),
    )

    conv3x3_relu_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W, C_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=1,
    )
    return y


@triton.jit
def linear_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ADD_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_rem = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
        b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Optional ReLU
    if ADD_RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def linear_triton(x, weight, bias, relu: bool):
    """
    x:      (M, K)
    weight: (N, K)  (PyTorch Linear weight)
    bias:   (N,)
    """
    assert x.ndim == 2
    M, K = x.shape
    N = weight.shape[0]

    # Use weight^T for better memory access (K, N)
    b = weight.t().contiguous()
    a = x.contiguous()

    c = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    linear_bias_relu_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ADD_RELU=relu,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Feature extractor (Conv + ReLU fused in Triton, pooling in PyTorch)
        # Block 1
        x = conv3x3_relu_triton(x, self.conv1_1.weight, self.conv1_1.bias)
        x = conv3x3_relu_triton(x, self.conv1_2.weight, self.conv1_2.bias)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        # Block 2
        x = conv3x3_relu_triton(x, self.conv2_1.weight, self.conv2_1.bias)
        x = conv3x3_relu_triton(x, self.conv2_2.weight, self.conv2_2.bias)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        # Block 3
        x = conv3x3_relu_triton(x, self.conv3_1.weight, self.conv3_1.bias)
        x = conv3x3_relu_triton(x, self.conv3_2.weight, self.conv3_2.bias)
        x = conv3x3_relu_triton(x, self.conv3_3.weight, self.conv3_3.bias)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        # Block 4
        x = conv3x3_relu_triton(x, self.conv4_1.weight, self.conv4_1.bias)
        x = conv3x3_relu_triton(x, self.conv4_2.weight, self.conv4_2.bias)
        x = conv3x3_relu_triton(x, self.conv4_3.weight, self.conv4_3.bias)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        # Block 5
        x = conv3x3_relu_triton(x, self.conv5_1.weight, self.conv5_1.bias)
        x = conv3x3_relu_triton(x, self.conv5_2.weight, self.conv5_2.bias)
        x = conv3x3_relu_triton(x, self.conv5_3.weight, self.conv5_3.bias)
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten
        x = torch.flatten(x, 1)

        # Classifier (Linear + bias + optional ReLU fused in Triton; dropout p=0 skipped)
        x = linear_triton(x, self.fc1.weight, self.fc1.bias, relu=True)
        x = linear_triton(x, self.fc2.weight, self.fc2.bias, relu=True)
        x = linear_triton(x, self.fc3.weight, self.fc3.bias, relu=False)

        return x
