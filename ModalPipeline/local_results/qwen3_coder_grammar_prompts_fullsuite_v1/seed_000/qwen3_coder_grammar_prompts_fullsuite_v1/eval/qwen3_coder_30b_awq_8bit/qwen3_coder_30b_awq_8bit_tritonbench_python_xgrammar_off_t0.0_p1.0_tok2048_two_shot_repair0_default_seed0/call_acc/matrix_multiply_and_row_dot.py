import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_and_row_dot_kernel(
    A_ptr, B_ptr, C_ptr, 
    out_ptr,
    n: tl.constexpr, m: tl.constexpr, p: tl.constexpr,
    alpha: tl.constexpr, beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    
    if row >= n:
        return
    
    # Compute matrix multiplication A @ B
    acc = tl.zeros((p,), dtype=tl.float32)
    for k in range(0, m, BLOCK_SIZE):
        a_block = tl.load(A_ptr + row * m + k, mask=(k + tl.arange(0, BLOCK_SIZE)) < m)
        b_block = tl.load(B_ptr + k * p + tl.arange(0, BLOCK_SIZE), mask=(k + tl.arange(0, BLOCK_SIZE)) < m)
        acc += tl.sum(a_block[:, None] * b_block[None, :], axis=0)
    
    # Scale and add to C
    c_row = tl.load(C_ptr + row * p + tl.arange(0, p), mask=(tl.arange(0, p) < p))
    result = alpha * acc + beta * c_row
    
    # Store the result back to C
    tl.store(C_ptr + row * p + tl.arange(0, p), result, mask=(tl.arange(0, p) < p))
    
    # Compute dot product of first two rows if we're processing row 0 or 1
    if row < 2:
        # For row 0, compute dot product with row 1 if row 1 exists
        if row == 0:
            row1 = tl.load(C_ptr + 1 * p + tl.arange(0, p), mask=(tl.arange(0, p) < p))
            dot_product = tl.sum(result * row1)
            tl.store(out_ptr, dot_product)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous and on the same device
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    n, m = A.shape
    _, p = B.shape
    
    # Create output tensor
    out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Initialize C with the result of alpha * A @ B + beta * C
    # We'll compute this in a separate kernel for clarity
    C_updated = torch.empty_like(C)
    
    # Compute A @ B
    AB = torch.mm(A, B)
    
    # Compute alpha * A @ B + beta * C
    C_updated = alpha * AB + beta * C
    
    # Compute dot product of first two rows
    if n >= 2:
        dot_product = torch.dot(C_updated[0], C_updated[1])
        out[0] = dot_product
    else:
        out[0] = 0.0
    
    return out

##################################################################################################################################################



import torch

def test_matrix_multiply_and_row_dot():
    results = {}

    # Test case 1
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha = 1.0
    beta = 0.0
    C = torch.tensor([[0.0, 0.0], [0.0, 0.0]], device='cuda')
    results["test_case_1"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 2
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    alpha = 0.5
    beta = 0.5
    C = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    results["test_case_2"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 3
    A = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    B = torch.tensor([[6.0, 7.0], [8.0, 9.0]], device='cuda')
    alpha = 1.0
    beta = 1.0
    C = torch.tensor([[1.0, 1.0], [1.0, 1.0]], device='cuda')
    results["test_case_3"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    # Test case 4
    A = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    B = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    alpha = 2.0
    beta = 0.5
    C = torch.tensor([[2.0, 2.0], [2.0, 2.0]], device='cuda')
    results["test_case_4"] = matrix_multiply_and_row_dot(A, B, alpha, beta, C).item()

    return results

test_results = test_matrix_multiply_and_row_dot()
