import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _qr_solve_kernel(
    A_ptr, b_ptr, x_ptr,
    M, N, K,
    stride_A_m, stride_A_n,
    stride_b_m, stride_b_k,
    stride_x_m, stride_x_k,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Initialize accumulator for the dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, N, BLOCK_SIZE_K):
        # Load A and b tiles
        a_tile = tl.load(
            A_ptr + pid_m * stride_A_m + k * stride_A_n,
            mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < M,
            other=0.0
        )
        b_tile = tl.load(
            b_ptr + k * stride_b_m + pid_n * stride_b_k,
            mask=(k + tl.arange(0, BLOCK_SIZE_K)) < N,
            other=0.0
        )
        
        # Compute dot product
        acc += tl.dot(a_tile, b_tile)
    
    # Store result
    tl.store(
        x_ptr + pid_m * stride_x_m + pid_n * stride_x_k,
        acc,
        mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < M,
    )

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are on the same device and have correct dtypes
    if A.device != b.device:
        raise ValueError("A and b must be on the same device")
    if A.dtype != b.dtype:
        raise ValueError("A and b must have the same dtype")
    
    # QR decomposition using torch.linalg.qr
    Q, R = torch.linalg.qr(A, mode='reduced')
    
    # Compute Q^T * b
    Q_T_b = torch.matmul(Q.t(), b)
    
    # Solve R * x = Q^T * b using triangular solver
    x = torch.triangular_solve(Q_T_b, R, upper=True)[0]
    
    return x
