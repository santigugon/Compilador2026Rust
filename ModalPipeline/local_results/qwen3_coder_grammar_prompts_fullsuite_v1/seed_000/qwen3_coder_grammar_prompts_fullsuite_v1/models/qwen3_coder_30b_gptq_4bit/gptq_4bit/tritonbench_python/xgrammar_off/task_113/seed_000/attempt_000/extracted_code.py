import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_mm_and_abs_sum_kernel(
    A_ptr, C_ptr, out_ptr,
    n, m,
    alpha, beta,
    BLOCK_SIZE_M=16, BLOCK_SIZE_N=16
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the symmetric matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, m, BLOCK_SIZE_N):
        a = tl.load(A_ptr + pid_m * BLOCK_SIZE_M + k * BLOCK_SIZE_N)
        b = tl.load(A_ptr + pid_n * BLOCK_SIZE_N + k * BLOCK_SIZE_M)
        acc += tl.dot(a, b)
    
    # Scale and accumulate
    c = tl.load(C_ptr + pid_m * BLOCK_SIZE_M + pid_n * BLOCK_SIZE_N)
    result = alpha * acc + beta * c
    
    # Store result
    tl.store(out_ptr + pid_m * BLOCK_SIZE_M + pid_n * BLOCK_SIZE_N, result)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    assert C.shape == (n, n), "C must have shape (n, n)"
    
    # Create output tensor
    out = torch.zeros_like(C)
    
    # Launch kernel
    grid = (triton.cdiv(n, 16), triton.cdiv(n, 16))
    symmetric_mm_and_abs_sum_kernel[grid](
        A, C, out,
        n, m,
        alpha, beta
    )
    
    # Return sum of absolute values
    return torch.sum(torch.abs(out))
