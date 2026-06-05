import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_matrix_vector_norm_kernel(
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
        a_row = tl.load(A_ptr + i * n + offsets, mask=mask)
        x_val = tl.load(x_ptr + i)
        y = y + alpha * a_row * x_val
    
    # Store result in y
    tl.store(y_ptr + offsets, y, mask=mask)
    
    # Compute norm
    if pid == 0:
        if p == 2.0:
            sum_sq = tl.sum(y * y)
            norm = tl.sqrt(sum_sq)
        elif p == 1.0:
            sum_abs = tl.sum(tl.abs(y))
            norm = sum_abs
        else:
            sum_pow = tl.sum(tl.power(tl.abs(y), p))
            norm = tl.power(sum_pow, 1.0 / p)
        tl.store(output_ptr, norm)

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be a square matrix"
    assert x.dim() == 1 and x.size(0) == A.size(0), "x must be a vector with same size as A's dimension"
    
    n = A.size(0)
    y = torch.zeros(n, dtype=torch.float32, device=A.device)
    
    # Initialize y with beta * y (initially zeros)
    y = y * beta
    
    # Compute alpha * A @ x
    for i in range(n):
        y += alpha * A[i, :] * x[i]
    
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
