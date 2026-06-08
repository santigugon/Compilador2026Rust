import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_mm_and_abs_sum_kernel(
    A_ptr, C_ptr, out_ptr,
    n, m,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the output matrix C = alpha * A @ A.T + beta * C
    # For each block of C, we compute a partial sum
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, m, BLOCK_SIZE_K):
        # Load A and A.T blocks
        a_block = tl.load(
            A_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * m +
            (k + tl.arange(0, BLOCK_SIZE_K)[None, :]),
            mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) < n and
                  (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) < m,
            other=0.0
        )
        
        b_block = tl.load(
            A_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * n +
            (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]),
            mask=(k + tl.arange(0, BLOCK_SIZE_K)[:, None]) < m and
                  (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) < n,
            other=0.0
        )
        
        # Compute partial dot product
        acc += tl.dot(a_block, b_block, allow_tf32=False)
    
    # Scale and accumulate
    c_block = tl.load(
        C_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * n +
        (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]),
        mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) < n and
              (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) < n,
        other=0.0
    )
    
    # Compute final result
    result = alpha * acc + beta * c_block
    
    # Store result back to C
    tl.store(
        C_ptr + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) * n +
        (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]),
        result,
        mask=(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None]) < n and
              (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) < n
    )
    
    # Compute sum of absolute values
    abs_result = tl.abs(result)
    sum_abs = tl.sum(abs_result)
    
    # Store the sum of absolute values
    tl.atomic_add(out_ptr, sum_abs)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are on the same device and have correct dtype
    device = A.device
    if device != C.device:
        raise ValueError("A and C must be on the same device")
    
    if A.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("A and C must be of type torch.float32")
    
    n, m = A.shape
    if C.shape != (n, n):
        raise ValueError("C must have shape (n, n) where A has shape (n, m)")
    
    # Create output tensor for sum of absolute values
    out = torch.zeros(1, dtype=torch.float32, device=device)
    
    # Launch kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    grid = (
        triton.cdiv(n, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N)
    )
    
    # Create a copy of C to avoid modifying the original
    C_copy = C.clone()
    
    # Launch kernel
    symmetric_mm_and_abs_sum_kernel[grid](
        A, C_copy, out,
        n, m,
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out
