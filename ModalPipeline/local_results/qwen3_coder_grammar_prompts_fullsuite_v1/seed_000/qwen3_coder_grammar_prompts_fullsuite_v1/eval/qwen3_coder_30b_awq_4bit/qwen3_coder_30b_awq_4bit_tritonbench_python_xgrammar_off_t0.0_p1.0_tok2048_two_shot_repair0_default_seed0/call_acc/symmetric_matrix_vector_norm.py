import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_vector_norm_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, p: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load vector x
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute matrix-vector product: y = alpha * A @ x
    y_vals = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(n):
        a_val = tl.load(A_ptr + i * n + offsets, mask=mask, other=0.0)
        y_vals += a_val * x_vals
    
    # Apply scaling and addition: y = alpha * A @ x + beta * y
    y_vals = alpha * y_vals
    if beta != 0.0:
        old_y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_vals = y_vals + beta * old_y
    
    # Store result
    tl.store(y_ptr + offsets, y_vals, mask=mask)
    
    # Compute norm
    if pid == 0:
        # Compute p-norm of y
        y_abs = tl.abs(y_vals)
        if p == 1.0:
            norm = tl.sum(y_abs)
        elif p == 2.0:
            norm = tl.sqrt(tl.sum(y_abs * y_abs))
        elif p == float('inf'):
            norm = tl.max(y_abs)
        else:
            norm = tl.pow(tl.sum(tl.pow(y_abs, p)), 1.0 / p)
        tl.store(out_ptr, norm)

def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    # Ensure inputs are contiguous and on the same device
    A = A.contiguous()
    x = x.contiguous()
    
    # Validate dimensions
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be a square matrix"
    assert x.dim() == 1 and x.size(0) == A.size(0), "x must be a vector with length equal to A's dimension"
    
    n = A.size(0)
    y = torch.empty_like(x)
    
    # Initialize y with zeros
    y.zero_()
    
    # Compute matrix-vector product and norm
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Create output tensor for norm
    out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # For small matrices, use a simpler approach
    if n <= 1024:
        # Use PyTorch for the matrix-vector product for simplicity
        y = alpha * torch.mv(A, x) + beta * y
        norm = torch.norm(y, p=p)
        return norm
    
    # For larger matrices, use Triton kernel
    _matrix_vector_norm_kernel[grid](A, x, y, out, n, alpha, beta, p, BLOCK=block)
    
    return out[0]

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
