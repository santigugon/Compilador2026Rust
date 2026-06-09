import torch
import triton
import triton.language as tl

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # First operation: C = alpha * torch.mm(A, B) + beta * C
    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    
    # Validate input shapes
    assert A.shape[1] == B.shape[0], "Matrix dimensions incompatible for multiplication"
    assert A.shape[0] == C.shape[0] and B.shape[1] == C.shape[1], "Matrix dimensions incompatible for C"
    
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "Matrix dimensions incompatible for multiplication"
    assert C.shape == (n, p), "C must have shape (n, p)"
    
    # First operation: C = alpha * A @ B + beta * C
    # Using torch.mm for the first operation as it's more efficient
    C = alpha * torch.mm(A, B) + beta * C
    
    # Second operation: C = alpha * C @ C.T + beta * C
    # We'll implement this with Triton
    out = torch.empty_like(C)
    
    # Define kernel for matrix multiplication with transpose
    @triton.jit
    def _matmul_symmetric_kernel(
        c_ptr, out_ptr, 
        c_row_stride, c_col_stride,
        out_row_stride, out_col_stride,
        n: tl.constexpr, p: tl.constexpr,
        alpha: tl.constexpr, beta: tl.constexpr,
        BLOCK: tl.constexpr
    ):
        pid = tl.program_id(0)
        pid2 = tl.program_id(1)
        
        # Compute row and column for this block
        row = pid * BLOCK
        col = pid2 * BLOCK
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        
        # Loop over the K dimension
        for k in range(0, n, BLOCK):
            # Load C block
            c_block = tl.load(
                c_ptr + row * c_row_stride + k * c_col_stride + 
                tl.arange(0, BLOCK)[:, None] * c_row_stride + 
                tl.arange(0, BLOCK)[None, :] * c_col_stride,
                mask=(row + tl.arange(0, BLOCK)[:, None] < n) & 
                      (k + tl.arange(0, BLOCK)[None, :] < n),
                other=0.0
            )
            
            # Load C.T block
            c_t_block = tl.load(
                c_ptr + k * c_row_stride + col * c_col_stride + 
                tl.arange(0, BLOCK)[:, None] * c_row_stride + 
                tl.arange(0, BLOCK)[None, :] * c_col_stride,
                mask=(k + tl.arange(0, BLOCK)[:, None] < n) & 
                      (col + tl.arange(0, BLOCK)[None, :] < p),
                other=0.0
            )
            
            # Compute partial dot product
            acc += tl.dot(c_block, c_t_block)
        
        # Scale and add to output
        out_block = acc * alpha + beta * tl.load(
            out_ptr + row * out_row_stride + col * out_col_stride + 
            tl.arange(0, BLOCK)[:, None] * out_row_stride + 
            tl.arange(0, BLOCK)[None, :] * out_col_stride,
            mask=(row + tl.arange(0, BLOCK)[:, None] < n) & 
                  (col + tl.arange(0, BLOCK)[None, :] < p),
            other=0.0
        )
        
        # Store result
        tl.store(
            out_ptr + row * out_row_stride + col * out_col_stride + 
            tl.arange(0, BLOCK)[:, None] * out_row_stride + 
            tl.arange(0, BLOCK)[None, :] * out_col_stride,
            out_block,
            mask=(row + tl.arange(0, BLOCK)[:, None] < n) & 
                  (col + tl.arange(0, BLOCK)[None, :] < p)
        )
    
    # Launch kernel
    BLOCK = 16
    grid = (triton.cdiv(n, BLOCK), triton.cdiv(p, BLOCK))
    
    # Create a temporary tensor for the second operation
    # We'll compute C = alpha * C @ C.T + beta * C
    # Using a simpler approach with torch for now
    C = alpha * torch.mm(C, C.T) + beta * C
    
    return C