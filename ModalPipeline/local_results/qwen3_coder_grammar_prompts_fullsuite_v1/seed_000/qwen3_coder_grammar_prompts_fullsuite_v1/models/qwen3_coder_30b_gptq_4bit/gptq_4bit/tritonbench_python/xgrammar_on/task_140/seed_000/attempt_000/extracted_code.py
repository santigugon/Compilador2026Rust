import torch
import triton
import triton.language as tl

def tril_mm_and_scale(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    assert A.shape[0] == A.shape[1], "Matrix A must be square"
    assert A.shape[1] == B.shape[0], "Matrix dimensions must be compatible for multiplication"
    
    n, p = A.shape[0], B.shape[1]
    
    # Create output tensor
    out = torch.empty(n, p, dtype=A.dtype, device=A.device)
    
    # Use Triton kernel for the operation
    _tril_mm_and_scale_kernel[1,](A, B, out, n, p, alpha, beta)
    
    return out

@triton.jit
def _tril_mm_and_scale_kernel(A_ptr, B_ptr, out_ptr, n: tl.constexpr, p: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr = 32):
    # Get block indices
    block_id_x = tl.program_id(0)
    block_id_y = tl.program_id(1)
    
    # Calculate block offsets
    start_x = block_id_x * BLOCK_SIZE
    start_y = block_id_y * BLOCK_SIZE
    
    # Load B block
    b_block = tl.load(B_ptr + tl.arange(0, BLOCK_SIZE) + start_y * B_ptr.stride(0))
    
    # Process each row in the block
    for i in range(BLOCK_SIZE):
        if start_x + i < n:
            # Load A row (lower triangular part)
            a_row = tl.load(A_ptr + (start_x + i) * A_ptr.stride(0) + tl.arange(0, p))
            
            # Compute dot product with B
            result = 0.0
            for k in range(p):
                # Only consider lower triangular part of A
                if k <= start_x + i:
                    result += a_row[k] * b_block[k]
                else:
                    break
            
            # Scale by alpha and beta
            result *= alpha
            result *= beta
            
            # Store result
            tl.store(out_ptr + (start_x + i) * out_ptr.stride(0) + start_y, result)