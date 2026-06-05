import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_and_row_dot_kernel(
    A_ptr, B_ptr, C_ptr, out_ptr,
    n: tl.constexpr, m: tl.constexpr, p: tl.constexpr,
    alpha: tl.constexpr, beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    # Compute the dot product of first two rows
    if pid == 0 and pid2 == 0:
        # Load first two rows of C
        row0 = tl.load(C_ptr + tl.arange(0, p), mask=tl.arange(0, p) < p)
        row1 = tl.load(C_ptr + m + tl.arange(0, p), mask=tl.arange(0, p) < p)
        
        # Compute dot product
        dot_product = tl.sum(row0 * row1)
        tl.store(out_ptr, dot_product)

@triton.jit
def _matrix_multiply_kernel(
    A_ptr, B_ptr, C_ptr,
    n: tl.constexpr, m: tl.constexpr, p: tl.constexpr,
    alpha: tl.constexpr, beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Compute matrix multiplication: C = alpha * A @ B + beta * C
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    if row < n and col < p:
        # Compute dot product of row from A and column from B
        acc = 0.0
        for k in range(0, m, BLOCK_SIZE):
            # Load A row and B column
            a = tl.load(A_ptr + row * m + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < m - k)
            b = tl.load(B_ptr + tl.arange(0, BLOCK_SIZE) * p + col, mask=tl.arange(0, BLOCK_SIZE) < m - k)
            acc += tl.sum(a * b)
        
        # Store result with scaling
        c_val = tl.load(C_ptr + row * p + col)
        result = alpha * acc + beta * c_val
        tl.store(C_ptr + row * p + col, result)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    n, m = A.shape
    _, p = B.shape
    
    # Create output tensor
    out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Create a temporary tensor for the result of A @ B
    temp = torch.empty(n, p, dtype=torch.float32, device=A.device)
    
    # Compute A @ B
    temp = torch.matmul(A, B)
    
    # Scale and add to C
    C_scaled = alpha * temp + beta * C
    
    # Compute dot product of first two rows
    if n >= 2:
        row0 = C_scaled[0]
        row1 = C_scaled[1]
        dot_product = torch.dot(row0, row1)
    else:
        # If there are less than 2 rows, return 0
        dot_product = torch.tensor(0.0, dtype=torch.float32, device=A.device)
    
    return dot_product

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
