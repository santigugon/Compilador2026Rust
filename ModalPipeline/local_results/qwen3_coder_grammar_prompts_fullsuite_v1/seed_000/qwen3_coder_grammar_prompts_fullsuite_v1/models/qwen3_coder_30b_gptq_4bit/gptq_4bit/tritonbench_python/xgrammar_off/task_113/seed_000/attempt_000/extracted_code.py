import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mm_and_abs_sum_kernel(
    a_ptr, c_ptr, out_ptr,
    n: tl.constexpr, m: tl.constexpr,
    alpha: tl.constexpr, beta: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the symmetric matrix multiplication part
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, m, BLOCK_N):
        # Load A block
        a_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        a_mask = a_offsets < n
        a_block = tl.load(a_ptr + a_offsets * m + k, mask=a_mask[:, None], other=0.0)
        
        # Load A.T block
        a_t_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a_t_mask = a_t_offsets < m
        a_t_block = tl.load(a_ptr + k + a_t_offsets * m, mask=a_t_mask[None, :], other=0.0)
        
        # Compute dot product
        acc += tl.dot(a_block, a_t_block)
    
    # Scale by alpha
    acc *= alpha
    
    # Load C and scale by beta
    c_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    c_mask = c_offsets < n
    c_block = tl.load(c_ptr + c_offsets * m + pid_n * BLOCK_N, mask=c_mask[:, None], other=0.0)
    c_block *= beta
    
    # Add to accumulator
    result = acc + c_block
    
    # Store result
    tl.store(out_ptr + c_offsets * m + pid_n * BLOCK_N, result, mask=c_mask[:, None])

def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    C = C.contiguous()
    
    n, m = A.shape
    
    # Create output tensor
    out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Define block size
    BLOCK_M = 32
    BLOCK_N = 32
    
    # Grid size
    grid_m = triton.cdiv(n, BLOCK_M)
    grid_n = triton.cdiv(m, BLOCK_N)
    grid = (grid_m, grid_n)
    
    # Allocate intermediate result tensor
    result = torch.empty_like(C)
    
    # Launch kernel
    _symmetric_mm_and_abs_sum_kernel[grid](
        A, C, result,
        n, m,
        alpha, beta,
        BLOCK_M, BLOCK_N
    )
    
    # Compute sum of absolute values
    return torch.sum(torch.abs(result))
