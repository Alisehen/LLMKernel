import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_gemm_mul_leaky_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    multiplier,
    negative_slope,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_TF32: tl.constexpr = True,
    num_stages: tl.constexpr = 2,
    num_warps: tl.constexpr = 8,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    
    GROUP_SIZE = tl.minimum(GROUP_M, num_pid_m)
    group_id = pid // GROUP_SIZE
    group_size = tl.minimum(num_pid_m - group_id * GROUP_SIZE, GROUP_SIZE)
    pid_in_group = pid % GROUP_SIZE
    
    first_pid_m = group_id * GROUP_SIZE
    grid_m = first_pid_m + pid_in_group
    grid_n = group_id % num_pid_n
    
    pid_m = grid_m
    pid_n = grid_n
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + 
                     tl.arange(0, BLOCK_K)[None, :] * stride_ak)
    b_ptrs = b_ptr + (tl.arange(0, BLOCK_K)[:, None] * stride_bk + 
                     offs_bn[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    a_mask = (offs_am[:, None] < M) & (tl.arange(0, BLOCK_K)[None, :] < K)
    b_mask = (tl.arange(0, BLOCK_K)[:, None] < K) & (offs_bn[None, :] < N)
    
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    
    num_blocks = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_blocks):
        next_k = k + 1
        has_next = next_k < num_blocks
        
        next_a = a
        next_b = b
        next_a_ptrs = a_ptrs
        next_b_ptrs = b_ptrs
        
        if has_next:
            next_a_ptrs = a_ptrs + BLOCK_K * stride_ak
            next_b_ptrs = b_ptrs + BLOCK_K * stride_bk
            
            next_a_mask = (offs_am[:, None] < M) & \
                         ((tl.arange(0, BLOCK_K) + next_k * BLOCK_K)[None, :] < K)
            next_b_mask = ((tl.arange(0, BLOCK_K) + next_k * BLOCK_K)[:, None] < K) & \
                         (offs_bn[None, :] < N)
            
            next_a = tl.load(next_a_ptrs, mask=next_a_mask, other=0.0)
            next_b = tl.load(next_b_ptrs, mask=next_b_mask, other=0.0)
        
        acc += tl.dot(a, b, allow_tf32=USE_TF32)
        
        if has_next:
            a = next_a
            b = next_b
            a_ptrs = next_a_ptrs
            b_ptrs = next_b_ptrs
    
    bias_ptrs = bias_ptr + offs_bn
    bias_mask = offs_bn < N
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    
    acc = tl.math.fma(acc, multiplier, bias[None, :] * multiplier)
    
    zero = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    negative_mask = acc < 0.0
    acc = tl.where(negative_mask, acc * negative_slope, acc)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + 
                      offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_gemm_mul_leaky_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    multiplier: float,
    negative_slope: float,
    configs=None
) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    
    c = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    x = x.contiguous()
    weight_t = weight.t().contiguous()
    bias = bias.contiguous()
    
    if configs is None:
        configs = [
            triton.Config({
                'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,
                'GROUP_M': 8, 'num_warps': 8, 'num_stages': 2
            }, num_stages=2, num_warps=8),
            triton.Config({
                'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,
                'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3
            }, num_stages=3, num_warps=8),
            triton.Config({
                'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32,
                'GROUP_M': 8, 'num_warps': 4, 'num_stages': 4
            }, num_stages=4, num_warps=4),
        ]
    
    def grid(meta):
        M, N = x.shape[0], weight.shape[0]
        grid_m = triton.cdiv(M, meta['BLOCK_M'])
        grid_n = triton.cdiv(N, meta['BLOCK_N'])
        
        GROUP_SIZE = min(meta['GROUP_M'], grid_m)
        num_groups = triton.cdiv(grid_m, GROUP_SIZE)
        total_blocks = num_groups * GROUP_SIZE * grid_n
        
        return (total_blocks,)
    
    config = configs[0]
    fused_gemm_mul_leaky_relu_kernel[grid](
        x, weight_t, bias, c,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        c.stride(0), c.stride(1),
        multiplier,
        negative_slope,
        **config.kwargs
    )
    
    return c


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        x = x.contiguous()
        return fused_gemm_mul_leaky_relu(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )
