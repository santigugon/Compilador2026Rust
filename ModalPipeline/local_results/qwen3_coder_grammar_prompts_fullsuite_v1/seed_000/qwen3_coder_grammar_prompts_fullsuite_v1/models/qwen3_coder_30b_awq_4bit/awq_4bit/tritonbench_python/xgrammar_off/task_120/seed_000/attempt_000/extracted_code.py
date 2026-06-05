import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_vector_dot_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Each program processes one row of A
    if pid < n:
        # Compute y = alpha * mv(A, x) + beta * y
        # For each row, compute dot product with x
        acc = tl.zeros((1,), dtype=tl.float32)
        for i in range(0, m, BLOCK_SIZE):
            # Load x with mask
            x_offsets = i + tl.arange(0, BLOCK_SIZE)
            x_mask = x_offsets < m
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            # Load A row with mask
            A_offsets = pid * m + x_offsets
            A_mask = x_offsets < m
            A_vals = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)
            
            # Compute partial dot product
            acc += tl.sum(A_vals * x_vals)
        
        # Compute y = alpha * mv(A, x) + beta * y
        y_val = tl.load(y_ptr + pid)
        y_new = alpha * acc + beta * y_val
        tl.store(y_ptr + pid, y_new)
        
        # Compute dot product of updated y with x
        dot_acc = tl.zeros((1,), dtype=tl.float32)
        for i in range(0, m, BLOCK_SIZE):
            x_offsets = i + tl.arange(0, BLOCK_SIZE)
            x_mask = x_offsets < m
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            y_offsets = pid * m + x_offsets
            y_mask = x_offsets < m
            y_vals = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)
            
            dot_acc += tl.sum(y_vals * x_vals)
        
        tl.store(out_ptr + pid, dot_acc)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are contiguous for efficient memory access
    A = A.contiguous()
    x = x.contiguous()
    y = y.contiguous()
    
    # Get dimensions
    n, m = A.shape
    assert x.shape == (m,), f"Expected x to have shape ({m},), got {x.shape}"
    assert y.shape == (n,), f"Expected y to have shape ({n},), got {y.shape}"
    
    # Create output tensor
    out = torch.empty(n, dtype=torch.float32, device=A.device)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (n,)
    
    # Create a temporary tensor for intermediate results
    temp_y = y.clone()
    
    # Compute the matrix-vector product and dot product
    _matrix_vector_dot_kernel[grid](A, x, temp_y, out, n, m, alpha, beta, BLOCK_SIZE)
    
    # Return the dot product result
    return out.sum()

# Alternative simpler implementation that's more accurate
@triton.jit
def _mv_dot_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid < n:
        # Compute matrix-vector product for this row
        acc = tl.zeros((1,), dtype=tl.float32)
        for i in range(0, m, BLOCK_SIZE):
            x_offsets = i + tl.arange(0, BLOCK_SIZE)
            x_mask = x_offsets < m
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            A_offsets = pid * m + x_offsets
            A_mask = x_offsets < m
            A_vals = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)
            
            acc += tl.sum(A_vals * x_vals)
        
        # Update y = alpha * mv(A, x) + beta * y
        y_val = tl.load(y_ptr + pid)
        y_new = alpha * acc + beta * y_val
        tl.store(y_ptr + pid, y_new)
        
        # Compute dot product of updated y with x
        dot_acc = tl.zeros((1,), dtype=tl.float32)
        for i in range(0, m, BLOCK_SIZE):
            x_offsets = i + tl.arange(0, BLOCK_SIZE)
            x_mask = x_offsets < m
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            y_offsets = pid * m + x_offsets
            y_mask = x_offsets < m
            y_vals = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)
            
            dot_acc += tl.sum(y_vals * x_vals)
        
        tl.store(out_ptr + pid, dot_acc)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are contiguous for efficient memory access
    A = A.contiguous()
    x = x.contiguous()
    y = y.contiguous()
    
    # Get dimensions
    n, m = A.shape
    assert x.shape == (m,), f"Expected x to have shape ({m},), got {x.shape}"
    assert y.shape == (n,), f"Expected y to have shape ({n},), got {y.shape}"
    
    # Create output tensor
    out = torch.empty(n, dtype=torch.float32, device=A.device)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (n,)
    
    # Copy y to avoid modifying original
    temp_y = y.clone()
    
    # Run kernel
    _mv_dot_kernel[grid](A, x, temp_y, out, n, m, alpha, beta, BLOCK_SIZE)
    
    # Return sum of all dot products
    return out.sum()
