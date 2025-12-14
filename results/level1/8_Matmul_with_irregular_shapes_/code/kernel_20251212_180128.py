import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride information
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Group size for cache optimization
    GROUP_M: tl.constexpr,
    # Number of warps
    num_warps: tl.constexpr = 4,
):
    """Optimized matrix multiplication kernel with improved grid mapping."""
    
    # 2D program ID grid for better parallel distribution
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Offsets for tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Load pointers
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Main K-loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load tiles with masking
        k_remaining = K - k * BLOCK_K
        
        # Load A tile with efficient masking
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B tile with efficient masking
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiply
        acc += tl.dot(a, b, allow_tf32=True)
        
        # Update pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton-optimized matrix multiplication with autotuned parameters."""
    # Check and ensure 2D tensors
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D tensors"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions for matrix multiplication"
    
    # Get dimensions
    M, K = a.shape
    _, N = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Get strides
    stride_am, stride_ak = a.stride()
    stride_bk, stride_bn = b.stride()
    stride_cm, stride_cn = c.stride()
    
    # Autotuned configurations for Ada Lovelace (RTX 4090)
    # Prioritize larger tile sizes for better SM utilization
    configs = [
        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_warps': 8, 'num_stages': 3},
        {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3},
        {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3},
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3},
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8, 'num_warps': 4, 'num_stages': 3},
    ]
    
    # Grid calculation function
    def grid(meta):
        num_pid_m = triton.cdiv(M, meta['BLOCK_M'])
        num_pid_n = triton.cdiv(N, meta['BLOCK_N'])
        num_pid_in_group = meta['GROUP_M'] * num_pid_n
        num_groups = triton.cdiv(num_pid_m, meta['GROUP_M'])
        return (num_groups * num_pid_in_group,)
    
    # Launch kernel with autotuning
    with torch.cuda.device(a.device):
        best_config = None
        best_time = float('inf')
        
        for config in configs:
            try:
                # Warmup
                matmul_kernel[grid(config)](a, b, c, M, N, K,
                                           stride_am, stride_ak, stride_bk, stride_bn,
                                           stride_cm, stride_cn,
                                           BLOCK_M=config['BLOCK_M'],
                                           BLOCK_N=config['BLOCK_N'],
                                           BLOCK_K=config['BLOCK_K'],
                                           GROUP_M=config['GROUP_M'],
                                           num_warps=config['num_warps'])
                
                # Benchmark
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                matmul_kernel[grid(config)](a, b, c, M, N, K,
                                           stride_am, stride_ak, stride_bk, stride_bn,
                                           stride_cm, stride_cn,
                                           BLOCK_M=config['BLOCK_M'],
                                           BLOCK_N=config['BLOCK_N'],
                                           BLOCK_K=config['BLOCK_K'],
                                           GROUP_M=config['GROUP_M'],
                                           num_warps=config['num_warps'])
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed_time = start_event.elapsed_time(end_event)
                
                if elapsed_time < best_time:
                    best_time = elapsed_time
                    best_config = config
                    
            except Exception as e:
                continue
        
        # Use best configuration for final execution
        if best_config is None:
            # Fallback to a safe configuration
            best_config = {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'num_warps': 4}
        
        matmul_kernel[grid(best_config)](a, b, c, M, N, K,
                                       stride_am, stride_ak, stride_bk, stride_bn,
                                       stride_cm, stride_cn,
                                       BLOCK_M=best_config['BLOCK_M'],
                                       BLOCK_N=best_config['BLOCK_N'],
                                       BLOCK_K=best_config['BLOCK_K'],
                                       GROUP_M=best_config['GROUP_M'],
                                       num_warps=best_config['num_warps'])
    
    return c

class ModelNew(nn.Module):
    """Optimized model using Triton for matrix multiplication."""
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Performs matrix multiplication of A and B using Triton kernels."""
        return triton_matmul(A, B)
