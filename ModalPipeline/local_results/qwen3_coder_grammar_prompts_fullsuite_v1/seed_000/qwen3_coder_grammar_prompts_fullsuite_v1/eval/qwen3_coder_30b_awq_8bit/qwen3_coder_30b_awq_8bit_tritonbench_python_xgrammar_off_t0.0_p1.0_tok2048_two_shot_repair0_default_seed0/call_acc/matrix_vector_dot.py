import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_vector_dot_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program processes one row of A
    if pid >= n:
        return
    
    # Load y element for this row
    y_elem = tl.load(y_ptr + pid)
    
    # Compute dot product of row pid of A with x
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
    
    # Compute y_new = alpha * (A @ x) + beta * y
    y_new = alpha * acc + beta * y_elem
    
    # Store the result back to y
    tl.store(y_ptr + pid, y_new)
    
    # For the final dot product, we need to accumulate y_new * x_i for all i
    # This requires a reduction across all elements, so we'll compute it in a separate kernel
    # But for now, we just store the intermediate y_new values

@triton.jit
def _dot_product_kernel(y_ptr, x_ptr, out_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute dot product of y and x
    acc = 0.0
    for i in range(0, n, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(y_vals * x_vals)
    
    # Store the final result
    tl.store(out_ptr, acc)

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Validate input shapes
    assert A.dim() == 2, "A must be a 2D tensor"
    assert x.dim() == 1, "x must be a 1D tensor"
    assert y.dim() == 1, "y must be a 1D tensor"
    assert A.shape[1] == x.shape[0], "A's second dimension must match x's dimension"
    assert A.shape[0] == y.shape[0], "A's first dimension must match y's dimension"
    
    n, m = A.shape
    
    # Create output tensor for the final dot product result
    out = torch.empty((), dtype=torch.float32, device=A.device)
    
    # First compute the matrix-vector product and update y in-place
    block_size = 256
    grid = (triton.cdiv(n, block_size),)
    
    # We need to compute y = alpha * (A @ x) + beta * y
    # This is a two-step process: compute A @ x, then update y
    
    # For the matrix-vector product, we'll use a different approach
    # Compute the updated y values
    _matrix_vector_dot_kernel[grid](A, x, y, out, n, m, alpha, beta, BLOCK_SIZE=block_size)
    
    # Now compute the dot product of updated y with x
    # We need to compute sum(y_new * x)
    dot_grid = (triton.cdiv(n, block_size),)
    _dot_product_kernel[dot_grid](y, x, out, n, BLOCK_SIZE=block_size)
    
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
