import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_vector_dot_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Each program processes one element of the output
    if pid == 0:
        # Compute y = alpha * A @ x + beta * y
        # Initialize accumulator for dot product
        dot_product = 0.0
        
        # Compute matrix-vector product A @ x
        for i in range(0, m, BLOCK):
            # Load x elements with masking
            x_offsets = i + tl.arange(0, BLOCK)
            x_mask = x_offsets < m
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            # Load A rows with masking
            for j in range(0, n, BLOCK):
                a_offsets = j + tl.arange(0, BLOCK)
                a_mask = a_offsets < n
                a_vals = tl.load(A_ptr + a_offsets * m + i, mask=a_mask, other=0.0)
                
                # Compute partial dot product
                partial = tl.sum(a_vals * x_vals)
                
                # Accumulate
                if i == 0:
                    # Initialize y values
                    y_vals = tl.load(y_ptr + a_offsets, mask=a_mask, other=0.0)
                    y_vals = alpha * partial + beta * y_vals
                    tl.store(y_ptr + a_offsets, y_vals, mask=a_mask)
                else:
                    # Accumulate partial results
                    pass  # This is a simplified approach
        
        # Compute dot product of updated y and x
        for i in range(0, n, BLOCK):
            y_offsets = i + tl.arange(0, BLOCK)
            y_mask = y_offsets < n
            y_vals = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)
            x_vals = tl.load(x_ptr + y_offsets, mask=y_mask, other=0.0)
            dot_product += tl.sum(y_vals * x_vals)
        
        # Store result
        tl.store(out_ptr, dot_product)

@triton.jit
def _mv_kernel(A_ptr, x_ptr, y_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Each program processes one row of A
    row = pid
    if row < n:
        # Compute A[row, :] @ x
        acc = 0.0
        for i in range(0, m, BLOCK):
            x_offsets = i + tl.arange(0, BLOCK)
            x_mask = x_offsets < m
            x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
            
            a_offsets = i + tl.arange(0, BLOCK)
            a_mask = a_offsets < m
            a_vals = tl.load(A_ptr + row * m + a_offsets, mask=a_mask, other=0.0)
            
            acc += tl.sum(a_vals * x_vals)
        
        # Update y[row] = alpha * acc + beta * y[row]
        y_val = tl.load(y_ptr + row)
        y_new = alpha * acc + beta * y_val
        tl.store(y_ptr + row, y_new)

@triton.jit
def _dot_kernel(y_ptr, x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Each program processes one element of the output
    if pid == 0:
        dot_product = 0.0
        for i in range(0, n, BLOCK):
            y_offsets = i + tl.arange(0, BLOCK)
            y_mask = y_offsets < n
            y_vals = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)
            x_vals = tl.load(x_ptr + y_offsets, mask=y_mask, other=0.0)
            dot_product += tl.sum(y_vals * x_vals)
        tl.store(out_ptr, dot_product)


def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Validate inputs
    assert A.dim() == 2, "A must be a 2D tensor"
    assert x.dim() == 1, "x must be a 1D tensor"
    assert y.dim() == 1, "y must be a 1D tensor"
    assert A.size(1) == x.size(0), "Matrix A column count must match vector x length"
    assert A.size(0) == y.size(0), "Matrix A row count must match vector y length"
    
    n, m = A.shape
    
    # Compute y = alpha * A @ x + beta * y
    block = 256
    grid = (triton.cdiv(n, block),)
    _mv_kernel[grid](A, x, y, n, m, alpha, beta, BLOCK=block)
    
    # Compute dot product of updated y and x
    out = torch.empty((), dtype=torch.float32, device=A.device)
    grid = (1,)
    _dot_kernel[grid](y, x, out, n, BLOCK=block)
    
    return out
##################################################################################################################################################



import torch
from torch import Tensor

def test_matrix_vector_dot():
    results = {}
    
    # Test case 1
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([0.0, 0.0], device='cuda')
    alpha = 1.0
    beta = 0.0
    results["test_case_1"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 2
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 1.0
    beta = 1.0
    results["test_case_2"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 3
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([2.0, 3.0], device='cuda')
    y = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 0.5
    beta = 0.5
    results["test_case_3"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    # Test case 4
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    y = torch.tensor([2.0, 2.0], device='cuda')
    alpha = 2.0
    beta = 0.5
    results["test_case_4"] = matrix_vector_dot(A, x, y, alpha, beta).item()
    
    return results

test_results = test_matrix_vector_dot()
