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
        for k in range(m):
            a_val = tl.load(A_ptr + k)  # First row of A
            b_val = tl.load(B_ptr + k * p)  # First row of B
            dot_result += a_val * b_val
            
        # Store the result
        tl.store(out_ptr, dot_result)

@triton.jit
def _matrix_multiply_kernel(
    A_ptr, B_ptr, C_ptr,
    n: tl.constexpr, m: tl.constexpr, p: tl.constexpr,
    alpha: tl.constexpr, beta: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, m, BLOCK_SIZE_K):
        a = tl.load(A_ptr + pid_m * m + k)
        b = tl.load(B_ptr + k * p + pid_n)
        acc += a * b
    
    # Scale and add to C
    c = tl.load(C_ptr + pid_m * p + pid_n)
    result = alpha * acc + beta * c
    
    # Store result
    tl.store(C_ptr + pid_m * p + pid_n, result)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    n, m = A.shape
    _, p = B.shape
    
    # Create output tensor
    out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Matrix multiplication part
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    grid_m = triton.cdiv(n, BLOCK_SIZE_M)
    grid_n = triton.cdiv(p, BLOCK_SIZE_N)
    grid = (grid_m, grid_n)
    
    # Launch matrix multiplication kernel
    _matrix_multiply_kernel[grid](
        A, B, C,
        n, m, p,
        alpha, beta,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    # Compute dot product of first two rows
    # For simplicity, we'll compute this on CPU for now
    # In a more optimized version, this would be done in Triton
    if n >= 2:
        first_row = C[0]
        second_row = C[1]
        dot_product = torch.dot(first_row, second_row)
        out[0] = dot_product.item()
    else:
        out[0] = 0.0
    
    return out
