import torch
import triton
import triton.language as tl

@triton.jit
def tril_mm_and_scale_kernel(
    A_ptr, B_ptr, out_ptr,
    n, p,
    alpha, beta,
    stride_a_row, stride_a_col,
    stride_b_row, stride_b_col,
    stride_out_row, stride_out_col,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    if pid >= num_blocks * num_blocks:
        return
    
    block_row = pid // num_blocks
    block_col = pid % num_blocks
    
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    for k in range(0, n, BLOCK_SIZE):
        a_block = tl.load(
            A_ptr + block_row * BLOCK_SIZE * stride_a_row + 
            k * stride_a_col,
            mask=(block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] < n) &
                  (k + tl.arange(0, BLOCK_SIZE)[None, :] < n),
            other=0.0
        )
        
        b_block = tl.load(
            B_ptr + k * stride_b_row + 
            block_col * BLOCK_SIZE * stride_b_col,
            mask=(k + tl.arange(0, BLOCK_SIZE)[:, None] < n) &
                  (block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :] < p),
            other=0.0
        )
        
        # Apply lower triangular mask to A
        mask = tl.arange(0, BLOCK_SIZE)[:, None] >= tl.arange(0, BLOCK_SIZE)[None, :]
        a_block = tl.where(mask, a_block, 0.0)
        
        acc += tl.dot(a_block, b_block)
    
    # Scale by alpha
    acc *= alpha
    
    # Scale by beta and store result
    out_block = acc * beta
    
    tl.store(
        out_ptr + block_row * BLOCK_SIZE * stride_out_row + 
        block_col * BLOCK_SIZE * stride_out_col,
        out_block,
        mask=(block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] < n) &
              (block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :] < p)
    )

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.dim() == 2 and B.dim() == 2
    assert A.shape[0] == A.shape[1] and A.shape[1] == B.shape[0]
    
    n, p = A.shape[0], B.shape[1]
    out = torch.empty(n, p, dtype=torch.float32, device=A.device)
    
    BLOCK_SIZE = 16
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks * num_blocks,)
    
    tril_mm_and_scale_kernel[grid](
        A, B, out,
        n, p,
        alpha, beta,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE
    )
    
    return out
