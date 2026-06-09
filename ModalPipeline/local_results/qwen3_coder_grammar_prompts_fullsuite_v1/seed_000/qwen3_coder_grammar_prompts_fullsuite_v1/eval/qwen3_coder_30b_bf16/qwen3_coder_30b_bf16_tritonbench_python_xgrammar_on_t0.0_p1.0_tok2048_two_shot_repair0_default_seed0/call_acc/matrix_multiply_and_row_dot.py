import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, n: tl.constexpr, m: tl.constexpr, p: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute matrix multiplication A @ B
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(p, BLOCK_SIZE)
    pid_n = pid % tl.cdiv(p, BLOCK_SIZE)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, m, BLOCK_SIZE):
        # Load tiles
        a_tile = tl.load(A_ptr + (tl.arange(0, BLOCK_SIZE)[:, None] * m + tl.arange(k, k + BLOCK_SIZE)[None, :]))
        b_tile = tl.load(B_ptr + (tl.arange(k, k + BLOCK_SIZE)[:, None] * p + tl.arange(0, BLOCK_SIZE)[None, :]))
        
        # Compute partial dot product
        acc += tl.dot(a_tile, b_tile)
    
    # Scale by alpha
    acc *= alpha
    
    # Load C and scale by beta
    c_tile = tl.load(C_ptr + (tl.arange(0, BLOCK_SIZE)[:, None] * p + tl.arange(0, BLOCK_SIZE)[None, :]))
    c_tile *= beta
    
    # Add to accumulator
    acc += c_tile
    
    # Store result
    tl.store(C_ptr + (tl.arange(0, BLOCK_SIZE)[:, None] * p + tl.arange(0, BLOCK_SIZE)[None, :]), acc)

@triton.jit
def _row_dot_kernel(C_ptr, out_ptr, n: tl.constexpr, p: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute dot product of first two rows
    acc = tl.zeros((), dtype=tl.float32)
    
    # Load first two rows
    row0 = tl.load(C_ptr + tl.arange(0, p))
    row1 = tl.load(C_ptr + p + tl.arange(0, p))
    
    # Compute dot product
    acc = tl.sum(row0 * row1)
    
    # Store result
    tl.store(out_ptr, acc)


def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    # Get dimensions
    n, m = A.shape
    m2, p = B.shape
    
    # Check dimensions
    assert m == m2, "Matrix dimensions do not match for multiplication"
    assert C.shape == (n, p), "C must have shape (n, p)"
    
    # Create output tensor
    out = torch.empty(1, dtype=torch.float32, device=A.device)
    
    # Compute matrix multiplication and update C
    BLOCK_SIZE = 16
    grid = (triton.cdiv(n, BLOCK_SIZE) * triton.cdiv(p, BLOCK_SIZE),)
    _matmul_kernel[grid](A, B, C, n, m, p, alpha, beta, BLOCK_SIZE)
    
    # Compute dot product of first two rows
    _row_dot_kernel[(1,)](C, out, n, p, BLOCK_SIZE)
    
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
