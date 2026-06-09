import torch
import triton
import triton.language as tl

@triton.jit
def symmetric_mv_norm_kernel(
    A_ptr, x_ptr, y_ptr, output_ptr,
    n,
    alpha,
    beta,
    p,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Initialize y with beta * y
    y = tl.load(y_ptr + offsets, mask=mask)
    y = y * beta
    
    # Compute alpha * A @ x
    for i in range(n):
        a_val = tl.load(A_ptr + i * n + offsets, mask=mask)
        x_val = tl.load(x_ptr + i)
        y = y + alpha * a_val * x_val
    
    # Store result back to y
    tl.store(y_ptr + offsets, y, mask=mask)
    
    # Compute norm
    if pid == 0:
        if p == 2.0:
            norm = tl.sqrt(tl.sum(y * y))
        elif p == 1.0:
            norm = tl.sum(tl.abs(y))
        else:
            norm = tl.pow(tl.sum(tl.pow(tl.abs(y), p)), 1.0 / p)
        tl.store(output_ptr, norm)

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    assert A.dim() == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"
    assert x.dim() == 1 and x.shape[0] == A.shape[0], "x must be a vector with length equal to A's dimension"
    assert A.is_contiguous() and x.is_contiguous(), "A and x must be contiguous"
    
    n = A.shape[0]
    y = torch.zeros_like(x)
    
    # Initialize y with beta * y
    y = y * beta
    
    # Compute alpha * A @ x
    for i in range(n):
        y += alpha * A[i] * x[i]
    
    # Compute norm
    if p == 2.0:
        norm = torch.sqrt(torch.sum(y * y))
    elif p == 1.0:
        norm = torch.sum(torch.abs(y))
    else:
        norm = torch.pow(torch.sum(torch.pow(torch.abs(y), p)), 1.0 / p)
    
    return norm

##################################################################################################################################################



import torch

def test_symmetric_matrix_vector_norm():
    results = {}

    # Test case 1: Basic test with default p value
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]], device='cuda')
    x = torch.tensor([1.0, 1.0], device='cuda')
    alpha = 1.0
    beta = 1.0
    results["test_case_1"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    # Test case 2: Different alpha and beta values
    alpha = 2.0
    beta = 0.5
    results["test_case_2"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    # Test case 3: Different p value (1-norm)
    alpha = 1.0
    beta = 1.0
    p = 1.0
    results["test_case_3"] = symmetric_matrix_vector_norm(A, x, alpha, beta, p).item()

    # Test case 4: Larger matrix and vector
    A = torch.tensor([[4.0, 1.0, 2.0], [1.0, 3.0, 1.0], [2.0, 1.0, 3.0]], device='cuda')
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    alpha = 1.5
    beta = 0.5
    results["test_case_4"] = symmetric_matrix_vector_norm(A, x, alpha, beta).item()

    return results

test_results = test_symmetric_matrix_vector_norm()
