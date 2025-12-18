import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused ReLU + Global Average Pool over spatial dims
# Optimized for contiguous NCHW tensors on Ada (e.g., RTX 4090)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit
def relu_global_avg_pool2d_kernel(
    x_ptr,        # *const fp16/fp32, [N, C, H*W] contiguous
    y_ptr,        # *fp16/fp32,       [N, C]     contiguous
    N: tl.int32,
    C: tl.int32,
    HW: tl.int32,
    BLOCK_HW: tl.constexpr,
):
    """
    Each program instance computes ReLU + global average over HW for one (n, c).
    No intermediate stores: all intermediates live in registers.
    """

    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Guard (in case grid is larger than (N, C))
    if (pid_n >= N) or (pid_c >= C):
        return

    # Flattened index for this (n, c) row in [N, C, HW] and [N, C]
    nc_idx = pid_n * C + pid_c
    x_base = nc_idx * HW      # start of this row over HW
    y_base = nc_idx           # single output element

    # Vector of HW indices handled by this program
    offs_hw = tl.arange(0, BLOCK_HW)
    acc_vec = tl.zeros((BLOCK_HW,), dtype=tl.float32)

    # Loop over spatial dimension in BLOCK_HW chunks
    for hw_start in range(0, HW, BLOCK_HW):
        idx = hw_start + offs_hw
        mask = idx < HW

        # Load contiguous slice x[n, c, idx], cast to fp32, apply ReLU
        x = tl.load(x_ptr + x_base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        x = tl.maximum(x, 0.0)

        # Accumulate per-lane partial sums (masked lanes contribute 0)
        acc_vec += tl.where(mask, x, 0.0)

    # Final reduction of vector accumulator to scalar
    acc = tl.sum(acc_vec, axis=0)

    # Compute mean (in fp32) and store once to global memory
    inv_HW = 1.0 / tl.full((), HW, dtype=tl.float32)
    avg = acc * inv_HW

    # Single final store (Triton will cast to y_ptr's dtype if needed)
    tl.store(y_ptr + y_base, avg)


# ---------------------------------------------------------------------------
# Wrapper: Python function to launch the fused kernel
# ---------------------------------------------------------------------------

def fused_relu_global_avg_pool2d(x: torch.Tensor) -> torch.Tensor:
    """
    Fuses ReLU + global average pooling over spatial dimensions.
    Input:  x of shape (N, C, H, W), contiguous NCHW
    Output: y of shape (N, C), same dtype/device as x
    """
    assert x.is_cuda, "Input must be on CUDA device"
    assert x.ndim == 4, "Expected input of shape (N, C, H, W)"

    # Ensure contiguous memory layout for optimal coalesced accesses
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W

    # Flatten spatial dims: [N, C, H, W] -> [N, C, HW]
    x_flat = x.view(N, C, HW)

    # Output tensor [N, C], contiguous
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)

    # 2D grid over (N, C); each program handles one (n, c) row
    grid = (N, C)

    relu_global_avg_pool2d_kernel[grid](
        x_flat,
        y,
        N,
        C,
        HW,
    )
    return y


# ---------------------------------------------------------------------------
# DenseNet-style building blocks
# ---------------------------------------------------------------------------

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Creates a single layer with BatchNorm, ReLU, Conv2D, and Dropout.
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Concatenated output tensor with shape (batch_size, num_output_features, height, width)
        """
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            # Concatenate along channel axis
            x = torch.cat(features, 1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        return self.transition(x)


# ---------------------------------------------------------------------------
# Model using the optimized fused Triton kernel
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        """
        :param growth_rate: The growth rate of the DenseNet (new features added per layer)
        :param num_classes: The number of output classes for classification
        """
        super(ModelNew, self).__init__()

        # Initial convolution and pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # DenseNet-style blocks
        num_features = 64
        block_layers = [6, 12, 24, 16]  # e.g., DenseNet-121

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, 3, height, width)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)

        # Optimized fused ReLU + global average pooling over spatial dims -> (N, C)
        x = fused_relu_global_avg_pool2d(x)

        # Linear classifier
        x = self.classifier(x)
        return x
