import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_and_row_dot_kernel(
    A_ptr, B_ptr, C_ptr,
    out_ptr,
    n: tl.constexpr, m: tl.constexpr, p: tl.constexpr,
    alpha: tl.constexpr, beta: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Compute matrix multiplication A @ B
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, m, BLOCK_SIZE_K):
        # Load A and B tiles
        a_tile = tl.load(A_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * m + 
                         (k + tl.arange(0, BLOCK_SIZE_K)[None, :]))
        b_tile = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * p + 
                         (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]))
        
        # Accumulate
        acc += tl.dot(a_tile, b_tile)
    
    # Scale and add to C
    c_tile = tl.load(C_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * p + 
                     (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]))
    
    # Apply alpha and beta
    result = alpha * acc + beta * c_tile
    
    # Store result
    tl.store(out_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * p + 
             (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]), result)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Validate input dimensions
    assert A.shape[1] == B.shape[0], "Incompatible dimensions for matrix multiplication"
    assert A.shape[0] == C.shape[0] and B.shape[1] == C.shape[1], "Incompatible dimensions for C"
    
    # Create output tensor
    out = torch.empty_like(C)
    
    # Get dimensions
    n, m, p = A.shape[0], A.shape[1], B.shape[1]
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Grid size
    grid_m = triton.cdiv(n, BLOCK_SIZE_M)
    grid_n = triton.cdiv(p, BLOCK_SIZE_N)
    grid = (grid_m, grid_n)
    
    # Launch kernel
    _matrix_multiply_and_row_dot_kernel[grid](
        A, B, C, out,
        n, m, p,
        alpha, beta,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    # Compute dot product of first two rows
    result = torch.dot(out[0], out[1])
    
    return result
