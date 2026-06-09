import torch
import triton
import triton.language as tl

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are on the same device and have correct dtypes
    assert A.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA"
    assert A.dtype == torch.float32 and B.dtype == torch.float32 and C.dtype == torch.float32, "All tensors must be float32"
    assert A.shape[1] == B.shape[0], "Matrix dimensions must be compatible for multiplication"
    
    n, m = A.shape
    m, p = B.shape
    assert C.shape == (n, p), "Matrix C must have shape (n, p)"
    
    # Create output tensor
    output = torch.empty(n, p, device=A.device, dtype=torch.float32)
    
    # Launch Triton kernel
    _matrix_multiply_and_row_dot_kernel[(n, p)](A, B, C, output, alpha, beta, n, m, p)
    
    # Return dot product of first two rows
    return torch.dot(output[0], output[1])

@triton.jit
def _matrix_multiply_and_row_dot_kernel(A, B, C, output, alpha, beta, n, m, p):
    # Get thread indices
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute matrix multiplication
    for i in range(m):
        a = tl.load(A + row * m + i)
        b = tl.load(B + i * p + col)
        acc += a * b
    
    # Scale and add to C
    c = tl.load(C + row * p + col)
    result = alpha * acc + beta * c
    
    # Store result
    tl.store(output + row * p + col, result)
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
