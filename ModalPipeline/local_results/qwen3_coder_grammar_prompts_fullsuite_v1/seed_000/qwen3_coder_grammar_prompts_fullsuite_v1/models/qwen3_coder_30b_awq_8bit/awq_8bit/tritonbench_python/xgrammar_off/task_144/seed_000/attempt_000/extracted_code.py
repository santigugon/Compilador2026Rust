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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the dot product of first two rows
    if pid_m == 0 and pid_n == 0:
        # Initialize accumulator for dot product
        dot_result = tl.zeros((1,), dtype=tl.float32)
        
        # Compute dot product of first two rows
        for k in range(0, p):
            # Load elements from first row of C
            c1 = tl.load(C_ptr + k, mask=k < p, other=0.0)
            # Load elements from second row of C
            c2 = tl.load(C_ptr + p + k, mask=k < p, other=0.0)
            # Accumulate dot product
            dot_result += c1 * c2
            
        # Store the result
        tl.store(out_ptr, dot_result[0])

@triton.jit
def _matrix_multiply_kernel(
    A_ptr, B_ptr, C_ptr,
    n: tl.constexpr, m: tl.constexpr, p: tl.constexpr,
    alpha: tl.constexpr, beta: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Initialize accumulator for matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute matrix multiplication
    for k in range(0, m, BLOCK_SIZE_K):
        # Load tiles
        a_tile = tl.load(A_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * m + 
                        (k + tl.arange(0, BLOCK_SIZE_K)[None, :]))
        b_tile = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * p + 
                        (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]))
        
        # Matrix multiplication
        acc += tl.dot(a_tile, b_tile)
    
    # Scale and add to C
    c_tile = tl.load(C_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * p + 
                    (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]))
    
    # Apply alpha and beta
    acc = alpha * acc + beta * c_tile
    
    # Store result
    tl.store(C_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * p + 
             (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]), acc)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    # Get dimensions
    n, m = A.shape
    m2, p = B.shape
    
    # Validate dimensions
    if m != m2:
        raise ValueError(f"Matrix dimensions incompatible: {A.shape} and {B.shape}")
    
    # Create output tensor
    out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Launch matrix multiplication kernel
    grid_m = triton.cdiv(n, BLOCK_SIZE_M)
    grid_n = triton.cdiv(p, BLOCK_SIZE_N)
    grid = (grid_m, grid_n)
    
    # First compute the matrix multiplication and update C
    _matrix_multiply_kernel[grid](
        A, B, C,
        n, m, p,
        alpha, beta,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    # Then compute the dot product of first two rows
    _matrix_multiply_and_row_dot_kernel[1, 1](
        A, B, C, out,
        n, m, p,
        alpha, beta,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return out
