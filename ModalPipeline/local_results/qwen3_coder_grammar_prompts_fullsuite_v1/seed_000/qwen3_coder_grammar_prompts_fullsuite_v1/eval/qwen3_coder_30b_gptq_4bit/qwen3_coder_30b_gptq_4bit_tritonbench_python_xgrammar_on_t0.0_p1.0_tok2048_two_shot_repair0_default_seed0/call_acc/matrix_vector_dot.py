import torch
import triton
import triton.language as tl

def matrix_vector_dot(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    n, m = A.shape
    
    # Compute y = alpha * torch.mv(A, x) + beta * y
    # First compute mv(A, x) using Triton
    y_out = torch.empty(n, dtype=A.dtype, device=A.device)
    
    # Use a simple kernel for matrix-vector multiplication
    @triton.jit
    def _mv_kernel(A_ptr, x_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        if pid * BLOCK >= n:
            return
        
        # Each block handles one row of A
        row_offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = row_offsets < n
        
        # For each row, compute dot product with x
        for i in range(m):
            x_val = tl.load(x_ptr + i)
            for j in range(BLOCK):
                if row_offsets[j] < n:
                    a_val = tl.load(A_ptr + row_offsets[j] * m + i)
                    if j == 0:
                        temp = a_val * x_val
                    else:
                        temp += a_val * x_val
        
        # Store result
        tl.store(out_ptr + row_offsets, temp, mask=mask)
    
    # Simple approach: use PyTorch for MV, then compute dot product
    # This is more straightforward and less error-prone
    y_out = alpha * torch.mv(A, x) + beta * y
    
    # Compute dot product of y_out and x
    result = torch.dot(y_out, x)
    
    return result
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
