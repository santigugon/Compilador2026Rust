import torch
import triton
import triton.language as tl

@triton.jit
def _mv_dot_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    # Each block processes one row of A
    row = pid
    if row >= n:
        return
    
    # Initialize accumulator for the dot product
    dot_prod = 0.0
    
    # Process each column of A with corresponding element of x
    for i in range(0, m, BLOCK_N):
        # Load x elements with masking
        x_offsets = i + tl.arange(0, BLOCK_N)
        x_mask = x_offsets < m
        x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        
        # Load A row elements with masking
        A_offsets = row * m + x_offsets
        A_mask = x_offsets < m
        A_vals = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)
        
        # Compute dot product of A[row, :] and x
        dot_prod += tl.sum(A_vals * x_vals)
    
    # Compute y = alpha * mv(A, x) + beta * y
    y_val = tl.load(y_ptr + row)
    y_new = alpha * dot_prod + beta * y_val
    tl.store(y_ptr + row, y_new)
    
    # Store the final dot product in out_ptr
    if pid == 0:
        tl.store(out_ptr, y_new)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Validate input dimensions
    assert A.dim() == 2, "A must be a 2D tensor"
    assert x.dim() == 1, "x must be a 1D tensor"
    assert y.dim() == 1, "y must be a 1D tensor"
    assert A.size(1) == x.size(0), "A's number of columns must match x's length"
    assert A.size(0) == y.size(0), "A's number of rows must match y's length"
    
    n, m = A.shape
    out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Use a simple approach for small matrices or fallback to PyTorch
    if n * m < 1024:
        # For small matrices, use PyTorch's native implementation
        y_new = alpha * torch.mv(A, x) + beta * y
        return torch.dot(y_new, x)
    
    # For larger matrices, use Triton kernel
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (n,)
    
    # Create a temporary tensor to store the result of the dot product
    temp_result = torch.empty_like(y)
    
    # Launch kernel
    _mv_dot_kernel[grid](A, x, y, out, n, m, alpha, beta, BLOCK_M, BLOCK_N)
    
    # Return the final dot product
    return out
