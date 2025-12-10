import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define Tensor Core optimized matrix multiplication kernel
kernel_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_SIZE 128
#define WARPS_PER_BLOCK 8

__global__ void matmul_tensorcore_kernel(
    const at::Half* __restrict__ A,
    const at::Half* __restrict__ B,
    at::Half* __restrict__ C,
    int N) {
    
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    
    // Warp layout: 2x4 warps per block
    const int warpM = warpId / 2;  // 0-3
    const int warpN = warpId % 2;  // 0-1
    
    // Each warp computes a 64x64 tile (4x4 WMMA tiles)
    const int warpRow = blockIdx.y * BLOCK_SIZE + warpM * 64;
    const int warpCol = blockIdx.x * BLOCK_SIZE + warpN * 64;
    
    // WMMA fragments for 4x4 tile grid
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[4][4];
    
    // Initialize accumulators to zero
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            nvcuda::wmma::fill_fragment(c_frag[i][j], __float2half(0.0f));
        }
    }
    
    // Main loop over K dimension
    for (int k = 0; k < N; k += WMMA_K) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            // Load A fragment (16x16 tile)
            int a_row = warpRow + i * WMMA_M;
            int a_col = k;
            
            if (a_row < N && a_col < N) {
                nvcuda::wmma::load_matrix_sync(a_frag, &A[a_row * N + a_col], N);
            }
            
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                // Load B fragment (16x16 tile)
                int b_row = k;
                int b_col = warpCol + j * WMMA_N;
                
                if (b_row < N && b_col < N) {
                    nvcuda::wmma::load_matrix_sync(b_frag, &B[b_row * N + b_col], N);
                }
                
                // Perform matrix multiply-accumulate
                nvcuda::wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
            }
        }
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int c_row = warpRow + i * WMMA_M;
            int c_col = warpCol + j * WMMA_N;
            
            if (c_row < N && c_col < N) {
                nvcuda::wmma::store_matrix_sync(&C[c_row * N + c_col], c_frag[i][j], N, nvcuda::wmma::mem_col_major);
            }
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matmul_tensorcore_kernel<<<grid, block>>>(
        reinterpret_cast<const at::Half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const at::Half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<at::Half*>(C.data_ptr<at::Half>()),
        N);
    
    return C;
}
"""

cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile custom kernel
matmul_tensorcore = load_inline(
    name="matmul_tensorcore",
    cpp_sources=cpp_source,
    cuda_sources=kernel_source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode", "arch=compute_89,code=sm_89"],
)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_fn = matmul_tensorcore
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Convert to half precision for Tensor Core acceleration
        A_fp16 = A.to(torch.float16)
        B_fp16 = B.to(torch.float16)
        
        # Perform matrix multiplication using custom kernel
        C_fp16 = self.matmul_fn.matmul_cuda(A_fp16, B_fp16)
        
        # Return as float32 for compatibility
        return C_fp16.to(torch.float32)

def get_inputs():
    N = 2048 * 2
    # Generate inputs on GPU directly
    A = torch.rand(N, N, dtype=torch.float32, device='cuda')
    B = torch.rand(N, N, dtype=torch.float32, device='cuda')
    return [A, B]

def get_init_inputs():
    return []