import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def linear_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ACT_RELU: tl.constexpr,
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
        k_remaining = K - k

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
            other=0.0,
        )

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Optional ReLU
    if ACT_RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_linear(x: torch.Tensor,
                 weight: torch.Tensor,
                 bias: torch.Tensor,
                 activation: str = "none") -> torch.Tensor:
    """
    High-performance Linear (+bias [+ReLU]) using Triton.
    x: [M, K]
    weight: [N, K]  (same layout as nn.Linear.weight)
    bias: [N]
    activation: "none" or "relu"
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    assert x.dtype == weight.dtype == bias.dtype == torch.float32

    M, K = x.shape
    N, Kw = weight.shape
    assert Kw == K, "Weight in_features must match input features"

    # We compute: out = x @ weight.T + bias
    # Arrange B = weight.T with shape [K, N]
    b_mat = weight.t().contiguous()

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    ACT_RELU = activation == "relu"

    linear_bias_relu_kernel[grid](
        x, b_mat, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        b_mat.stride(0), b_mat.stride(1),
        out.stride(0), out.stride(1),
        ACT_RELU,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        num_warps=8, num_stages=3,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Triton-accelerated VGG16:
        - Convolutional feature extractor as in the original model (PyTorch Conv2d/MaxPool2d)
        - Classifier (three Linear layers) implemented with high-performance Triton GEMMs
        """
        super(ModelNew, self).__init__()

        # Feature extractor: identical to original VGG16 definition
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classifier weights as Parameters, to be used by Triton linear kernels
        in_features = 512 * 7 * 7
        hidden = 4096

        self.fc1_weight = nn.Parameter(torch.randn(hidden, in_features))
        self.fc1_bias = nn.Parameter(torch.randn(hidden))

        self.fc2_weight = nn.Parameter(torch.randn(hidden, hidden))
        self.fc2_bias = nn.Parameter(torch.randn(hidden))

        self.fc3_weight = nn.Parameter(torch.randn(num_classes, hidden))
        self.fc3_bias = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        """
        Forward pass:
        - Convolutional backbone via PyTorch
        - Classifier via Triton fused Linear(+Bias[+ReLU]) kernels
        """
        x = self.features(x)
        x = torch.flatten(x, 1)  # [B, 512*7*7]

        # Ensure tensors are float32 on CUDA for Triton
        if not x.is_cuda:
            x = x.cuda()
        x = x.to(torch.float32)
        self.fc1_weight.data = self.fc1_weight.data.to(torch.float32).cuda()
        self.fc1_bias.data = self.fc1_bias.data.to(torch.float32).cuda()
        self.fc2_weight.data = self.fc2_weight.data.to(torch.float32).cuda()
        self.fc2_bias.data = self.fc2_bias.data.to(torch.float32).cuda()
        self.fc3_weight.data = self.fc3_weight.data.to(torch.float32).cuda()
        self.fc3_bias.data = self.fc3_bias.data.to(torch.float32).cuda()

        x = fused_linear(x, self.fc1_weight, self.fc1_bias, activation="relu")
        x = fused_linear(x, self.fc2_weight, self.fc2_bias, activation="relu")
        x = fused_linear(x, self.fc3_weight, self.fc3_bias, activation="none")
        return x
