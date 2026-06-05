import torch
import triton
import triton.language as tl

@triton.jit
def _solve_and_add_scaled_vector_kernel(
    A_ptr, b_ptr, y_ptr, out_ptr, 
    n: tl.constexpr, k: tl.constexpr, 
    alpha: tl.constexpr,
    A_stride_0: tl.constexpr, A_stride_1: tl.constexpr,
    b_stride_0: tl.constexpr,
    y_stride_0: tl.constexpr,
    out_stride_0: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if k == 1:
        # For single column case
        for i in range(0, n, BLOCK):
            # Load A block
            a_offsets = i + tl.arange(0, BLOCK)
            a_mask = a_offsets < n
            a_block = tl.load(A_ptr + a_offsets * A_stride_0 + a_offsets * A_stride_1, mask=a_mask[:, None] & a_mask[None, :], other=0.0)
            
            # Load b block
            b_offsets = i + tl.arange(0, BLOCK)
            b_mask = b_offsets < n
            b_block = tl.load(b_ptr + b_offsets * b_stride_0, mask=b_mask, other=0.0)
            
            # Solve triangular system
            for j in range(i, min(i + BLOCK, n)):
                if j < n:
                    # Forward substitution for upper triangular matrix
                    if j > i:
                        b_block = b_block - a_block[:, j] * b_block[j]
                    if j == i:
                        b_block = b_block / a_block[j, j]
            
            # Store result
            out_offsets = i + tl.arange(0, BLOCK)
            out_mask = out_offsets < n
            tl.store(out_ptr + out_offsets * out_stride_0, b_block, mask=out_mask)
    else:
        # For multiple columns case
        for i in range(0, n, BLOCK):
            # Load A block
            a_offsets = i + tl.arange(0, BLOCK)
            a_mask = a_offsets < n
            a_block = tl.load(A_ptr + a_offsets * A_stride_0 + a_offsets * A_stride_1, mask=a_mask[:, None] & a_mask[None, :], other=0.0)
            
            # Load b block
            b_offsets = i + tl.arange(0, BLOCK)
            b_mask = b_offsets < n
            b_block = tl.load(b_ptr + b_offsets * b_stride_0, mask=b_mask, other=0.0)
            
            # Solve triangular system
            for j in range(i, min(i + BLOCK, n)):
                if j < n:
                    # Forward substitution for upper triangular matrix
                    if j > i:
                        b_block = b_block - a_block[:, j] * b_block[j]
                    if j == i:
                        b_block = b_block / a_block[j, j]
            
            # Store result
            out_offsets = i + tl.arange(0, BLOCK)
            out_mask = out_offsets < n
            tl.store(out_ptr + out_offsets * out_stride_0, b_block, mask=out_mask)

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Check if A is upper triangular
    if not torch.is_triu(A):
        raise ValueError("Matrix A must be upper triangular")
    
    # Check dimensions
    n, k = b.shape[0], 1
    if len(b.shape) > 1:
        k = b.shape[1]
    
    if y.shape[0] != n:
        raise ValueError("Vector y must have the same number of rows as matrix A")
    
    # Create output tensor
    out = torch.empty_like(b)
    
    # Solve triangular system using torch.linalg.solve_triangular
    x = torch.linalg.solve_triangular(A, b, upper=True)
    
    # Add scaled vector y to solution x
    out = x + alpha * y
    
    return out
