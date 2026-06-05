import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mm_and_abs_sum_kernel(A_ptr, C_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the block indices
    block_m_start = pid_m * BLOCK_M
    block_n_start = pid_n * BLOCK_N
    
    # Initialize accumulator for the result
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute the symmetric matrix multiplication: alpha * A @ A.T
    for k in range(0, m, BLOCK_M):
        # Load A block
        a_offsets = block_m_start * m + k + tl.arange(0, BLOCK_M)[:, None] * m + tl.arange(0, BLOCK_M)[None, :]
        a_mask = (block_m_start + tl.arange(0, BLOCK_M)[:, None] < n) & (k + tl.arange(0, BLOCK_M)[None, :] < m)
        a_block = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Load A.T block (transposed)
        a_t_offsets = k * n + block_n_start + tl.arange(0, BLOCK_M)[:, None] * n + tl.arange(0, BLOCK_N)[None, :]
        a_t_mask = (k + tl.arange(0, BLOCK_M)[:, None] < m) & (block_n_start + tl.arange(0, BLOCK_N)[None, :] < n)
        a_t_block = tl.load(A_ptr + a_t_offsets, mask=a_t_mask, other=0.0)
        
        # Compute dot product
        acc += tl.dot(a_block, a_t_block, allow_tf32=False)
    
    # Scale by alpha
    acc *= alpha
    
    # Load C and scale it by beta
    c_offsets = block_m_start * n + block_n_start + tl.arange(0, BLOCK_M)[:, None] * n + tl.arange(0, BLOCK_N)[None, :]
    c_mask = (block_m_start + tl.arange(0, BLOCK_M)[:, None] < n) & (block_n_start + tl.arange(0, BLOCK_N)[None, :] < n)
    c_block = tl.load(C_ptr + c_offsets, mask=c_mask, other=0.0)
    c_scaled = c_block * beta
    
    # Add the results
    result = acc + c_scaled
    
    # Compute sum of absolute values
    abs_result = tl.abs(result)
    sum_abs = tl.sum(abs_result)
    
    # Store the sum
    tl.store(out_ptr, sum_abs, mask=True)

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    C = C.contiguous()
    
    # Get dimensions
    n, m = A.shape
    assert C.shape == (n, n), "C must have shape (n, n)"
    
    # Initialize output tensor
    out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Define block size
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Compute grid size
    grid_m = triton.cdiv(n, BLOCK_M)
    grid_n = triton.cdiv(n, BLOCK_N)
    grid = (grid_m, grid_n)
    
    # Launch kernel
    _symmetric_mm_and_abs_sum_kernel[grid](
        A, C, out, n, m, alpha, beta, BLOCK_M, BLOCK_N
    )
    
    return out
