import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_vector_dot_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Compute the matrix-vector product first
    # Each thread computes one element of the result vector
    if pid < n:
        acc = 0.0
        for i in range(0, m, BLOCK_SIZE):
            # Load x elements with masking
            x_offsets = i + tl.arange(0, BLOCK_SIZE)
            x_mask = x_offsets < m
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            # Load A row elements with masking
            A_offsets = pid * m + x_offsets
            A_mask = x_offsets < m
            A_vals = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)
            
            # Compute partial dot product
            acc += tl.sum(A_vals * x_vals)
        
        # Store the matrix-vector product result
        y_new = alpha * acc + beta * tl.load(y_ptr + pid)
        tl.store(y_ptr + pid, y_new)
    
    # Now compute the dot product of updated y and x
    if pid == 0:
        dot_prod = 0.0
        for i in range(0, n, BLOCK_SIZE):
            y_offsets = i + tl.arange(0, BLOCK_SIZE)
            y_mask = y_offsets < n
            y_vals = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)
            
            x_offsets = y_offsets
            x_mask = y_offsets < m
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            dot_prod += tl.sum(y_vals * x_vals)
        
        tl.store(out_ptr, dot_prod)

@triton.jit
def _matrix_vector_dot_kernel_simple(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr):
    # Simple version: compute matrix-vector product and dot product in separate steps
    pid = tl.program_id(0)
    
    # Compute matrix-vector product: y = alpha * A @ x + beta * y
    if pid < n:
        acc = 0.0
        for i in range(m):
            acc += tl.load(A_ptr + pid * m + i) * tl.load(x_ptr + i)
        y_new = alpha * acc + beta * tl.load(y_ptr + pid)
        tl.store(y_ptr + pid, y_new)
    
    # Compute dot product of updated y and x
    if pid == 0:
        dot_prod = 0.0
        for i in range(n):
            dot_prod += tl.load(y_ptr + i) * tl.load(x_ptr + i)
        tl.store(out_ptr, dot_prod)

@triton.jit
def _matrix_vector_dot_kernel_optimized(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Optimized version with shared memory for x
    pid = tl.program_id(0)
    
    # Compute matrix-vector product: y = alpha * A @ x + beta * y
    if pid < n:
        acc = 0.0
        for i in range(0, m, BLOCK_SIZE):
            x_offsets = i + tl.arange(0, BLOCK_SIZE)
            x_mask = x_offsets < m
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            A_offsets = pid * m + x_offsets
            A_mask = x_offsets < m
            A_vals = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)
            
            acc += tl.sum(A_vals * x_vals)
        
        y_new = alpha * acc + beta * tl.load(y_ptr + pid)
        tl.store(y_ptr + pid, y_new)
    
    # Compute dot product of updated y and x
    if pid == 0:
        dot_prod = 0.0
        for i in range(0, n, BLOCK_SIZE):
            y_offsets = i + tl.arange(0, BLOCK_SIZE)
            y_mask = y_offsets < n
            y_vals = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)
            
            x_offsets = y_offsets
            x_mask = y_offsets < m
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            dot_prod += tl.sum(y_vals * x_vals)
        
        tl.store(out_ptr, dot_prod)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Validate inputs
    assert A.dim() == 2, "A must be a 2D tensor"
    assert x.dim() == 1, "x must be a 1D tensor"
    assert y.dim() == 1, "y must be a 1D tensor"
    assert A.size(1) == x.size(0), "A's second dimension must match x's size"
    assert A.size(0) == y.size(0), "A's first dimension must match y's size"
    
    n, m = A.shape
    
    # Create output tensor
    out = torch.empty((), dtype=torch.float32, device=A.device)
    
    # Launch kernel
    block_size = 256
    grid_size = max(1, triton.cdiv(n, block_size))
    
    # Use the optimized kernel
    _matrix_vector_dot_kernel_optimized[grid_size](A, x, y, out, n, m, alpha, beta, block_size)
    
    return out