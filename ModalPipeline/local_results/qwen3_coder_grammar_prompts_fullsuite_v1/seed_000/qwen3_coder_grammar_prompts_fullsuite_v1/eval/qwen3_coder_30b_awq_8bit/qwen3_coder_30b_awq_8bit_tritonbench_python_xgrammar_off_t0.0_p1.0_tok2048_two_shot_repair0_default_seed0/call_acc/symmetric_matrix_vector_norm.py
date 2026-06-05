import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_mv_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load x values
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute matrix-vector product for symmetric matrix
    result = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(n):
        a_val = tl.load(A_ptr + i * n + offsets, mask=mask, other=0.0)
        result += a_val * x_vals
    
    # Scale and accumulate with beta * y
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = alpha * result + beta * y_vals
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _norm_kernel(y_ptr, out_ptr, n: tl.constexpr, p: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load y values
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute |y|^p
    if p == 1.0:
        abs_vals = tl.abs(y_vals)
    elif p == 2.0:
        abs_vals = y_vals * y_vals
    else:
        abs_vals = tl.power(tl.abs(y_vals), p)
    
    # Reduce across the block
    result = tl.sum(abs_vals, axis=0)
    
    # Store partial result
    tl.store(out_ptr + pid, result, mask=pid < triton.cdiv(n, BLOCK))

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    # Validate inputs
    assert A.dim() == 2 and A.shape[0] == A.shape[1], "A must be a square matrix"
    assert x.dim() == 1 and x.shape[0] == A.shape[0], "x must be a vector with length matching A's dimension"
    
    n = A.shape[0]
    
    # Initialize output vector y
    y = torch.empty_like(x)
    
    # Compute y = alpha * mv(A, x) + beta * y
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # For small matrices, use PyTorch directly for better numerical stability
    if n <= 1024:
        y = alpha * torch.mv(A, x) + beta * y
    else:
        # Use Triton kernel for larger matrices
        _symmetric_mv_kernel[grid](A, x, y, y, n, alpha, beta, BLOCK=block)
    
    # Compute norm of y
    if p == 1.0:
        norm = torch.sum(torch.abs(y))
    elif p == 2.0:
        norm = torch.norm(y, p=2.0)
    else:
        norm = torch.sum(torch.abs(y) ** p) ** (1.0 / p)
    
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
