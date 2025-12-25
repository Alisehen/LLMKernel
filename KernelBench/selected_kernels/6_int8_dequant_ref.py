import torch
import torch.nn as nn

class Model(nn.Module):
    """
    INT8 MatMul with Row-wise Dequantization - PyTorch Reference Implementation

    Performs quantized matrix multiplication with per-row quantization:
    1. Quantize: FP -> INT8 with per-row scale
    2. Compute: INT8 @ INT8 -> INT32
    3. Dequantize: INT32 -> FP with row-wise scales

    Formula:
        output = (A_int8 @ B_int8) * scale_x * scale_w / (127 * 127)

    Where:
        - A_int8: Input activations quantized to INT8 [M, K]
        - B_int8: Weight matrix quantized to INT8 [K, N]
        - scale_x: Per-row scale for input [M]
        - scale_w: Per-column scale for weight [N]

    Used in: INT8 inference optimization for LLMs
    """
    def __init__(self, in_features=2048, out_features=2048):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Quantized weight matrix (INT8)
        self.weight_int8 = nn.Parameter(
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8),
            requires_grad=False
        )

        # Per-column scale for weights
        self.scale_w = nn.Parameter(
            torch.randn(out_features, dtype=torch.float32).abs() * 0.01,
            requires_grad=False
        )

        # Optional bias
        self.bias = nn.Parameter(
            torch.randn(out_features, dtype=torch.float16) * 0.01,
            requires_grad=False
        )

    def forward(self, x: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        """
        Perform INT8 matrix multiplication with dequantization.

        Args:
            x (torch.Tensor): Quantized input of shape (M, K), dtype=int8
            scale_x (torch.Tensor): Per-row scale for input of shape (M,), dtype=float32

        Returns:
            torch.Tensor: Dequantized output of shape (M, out_features), dtype=float16
        """
        M, K = x.shape

        # 1. INT8 @ INT8 -> INT32
        result_int32 = torch.matmul(
            x.float(),
            self.weight_int8.t().float()
        )  # [M, out_features]

        # 2. Dequantize: apply scales
        divfactor = 1.0 / (127.0 * 127.0)

        # scale_x: [M] -> [M, 1]
        # scale_w: [out_features] -> [1, out_features]
        # result_int32: [M, out_features]
        output = self.scale_w[None, :] * (scale_x[:, None] * (result_int32 * divfactor))

        # 3. Add bias
        output = output + self.bias[None, :]

        return output.to(torch.float16)


# Default configuration
M = 128
IN_FEATURES = 2048
OUT_FEATURES = 2048

def get_inputs():
    """
    Generate test inputs for INT8 matmul with dequantization.

    Returns:
        List containing:
            - x_int8: Quantized input [M, IN_FEATURES], dtype=int8
            - scale_x: Per-row scale [M], dtype=float32
    """
    # Generate random INT8 input
    x_int8 = torch.randint(-128, 127, (M, IN_FEATURES), dtype=torch.int8)

    # Generate per-row scales (simulating quantization scales)
    scale_x = torch.randn(M, dtype=torch.float32).abs() * 0.01

    return [x_int8, scale_x]

def get_init_inputs():
    """
    Get initialization parameters for Model.

    Returns:
        List containing [in_features, out_features]
    """
    return [IN_FEATURES, OUT_FEATURES]
