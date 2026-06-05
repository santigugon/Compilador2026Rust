import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_mm_and_abs_sum_kernel(
    A_ptr, C_ptr, 
    n, m,
    alpha, beta,
    C_sum_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the block indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Shared memory for tiles
    a_tile = tl.shared.load(A_ptr + m_start * m + tl.arange(0, BLOCK_SIZE_M)[:, None] * m + tl.arange(0, BLOCK_SIZE_N)[None, :])
    c_tile = tl.shared.load(C_ptr + m_start * m + tl.arange(0, BLOCK_SIZE_M)[:, None] * m + tl.arange(0, BLOCK_SIZE_N)[None, :])
    
    # Compute partial dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, m, BLOCK_SIZE_N):
        a_block = tl.load(A_ptr + m_start * m + tl.arange(0, BLOCK_SIZE_M)[:, None] * m + k + tl.arange(0, BLOCK_SIZE_N)[None, :])
        b_block = tl.load(A_ptr + k * m + tl.arange(0, BLOCK_SIZE_M)[:, None] * m + tl.arange(0, BLOCK_SIZE_N)[None, :])
        acc += tl.dot(a_block, b_block.T)
    
    # Scale and accumulate
    acc = alpha * acc + beta * c_tile
    
    # Store result back to C
    tl.store(C_ptr + m_start * m + tl.arange(0, BLOCK_SIZE_M)[:, None] * m + tl.arange(0, BLOCK_SIZE_N)[None, :], acc)
    
    # Compute sum of absolute values
    abs_acc = tl.abs(acc)
    sum_abs = tl.sum(abs_acc)
    tl.atomic_add(C_sum_ptr, sum_abs)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2
    assert C.dim() == 2
    assert A.shape == C.shape
    assert A.is_contiguous()
    assert C.is_contiguous()
    
    n, m = A.shape
    
    # Allocate output tensor for sum
    C_sum = torch.zeros(1, dtype=torch.float32, device=A.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    
    grid = (triton.cdiv(n, BLOCK_SIZE_M), triton.cdiv(m, BLOCK_SIZE_N))
    
    symmetric_mm_and_abs_sum_kernel[grid](
        A_ptr=A.data_ptr(),
        C_ptr=C.data_ptr(),
        n=n,
        m=m,
        alpha=alpha,
        beta=beta,
        C_sum_ptr=C_sum.data_ptr(),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return C_sum
