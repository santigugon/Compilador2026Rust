import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, out_ptr, 
                   n: tl.constexpr, m: tl.constexpr, p: tl.constexpr,
                   alpha: tl.constexpr, beta: tl.constexpr,
                   stride_a_row: tl.constexpr, stride_a_col: tl.constexpr,
                   stride_b_row: tl.constexpr, stride_b_col: tl.constexpr,
                   stride_c_row: tl.constexpr, stride_c_col: tl.constexpr,
                   stride_out_row: tl.constexpr, stride_out_col: tl.constexpr,
                   BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    # Compute output indices
    row = pid
    col = pid2
    
    if row < n and col < p:
        # Compute dot product for C = alpha * A @ B + beta * C
        acc = 0.0
        for k in range(m):
            a = tl.load(A_ptr + row * stride_a_row + k * stride_a_col)
            b = tl.load(B_ptr + k * stride_b_row + col * stride_b_col)
            acc += a * b
        
        # Load current C value
        c_val = tl.load(C_ptr + row * stride_c_row + col * stride_c_col)
        
        # Compute new C value
        new_c = alpha * acc + beta * c_val
        
        # Store result in C
        tl.store(C_ptr + row * stride_c_row + col * stride_c_col, new_c)
        
        # Store intermediate result in out for next operation
        tl.store(out_ptr + row * stride_out_row + col * stride_out_col, new_c)

@triton.jit
def _symmetric_update_kernel(C_ptr, out_ptr, 
                            n: tl.constexpr, p: tl.constexpr,
                            alpha: tl.constexpr, beta: tl.constexpr,
                            stride_c_row: tl.constexpr, stride_c_col: tl.constexpr,
                            stride_out_row: tl.constexpr, stride_out_col: tl.constexpr,
                            BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    row = pid
    col = pid2
    
    if row < n and col < p:
        # Compute dot product for C = alpha * C @ C.T + beta * C
        acc = 0.0
        for k in range(p):
            c1 = tl.load(C_ptr + row * stride_c_row + k * stride_c_col)
            c2 = tl.load(C_ptr + col * stride_c_row + k * stride_c_col)
            acc += c1 * c2
        
        # Load current C value
        c_val = tl.load(C_ptr + row * stride_c_row + col * stride_c_col)
        
        # Compute new C value
        new_c = alpha * acc + beta * c_val
        
        # Store result
        tl.store(out_ptr + row * stride_out_row + col * stride_out_col, new_c)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # Ensure inputs are contiguous and on the same device
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    # Get dimensions
    n, m = A.shape
    m2, p = B.shape
    n2, p2 = C.shape
    
    # Validate dimensions
    assert m == m2, f"Matrix A column count ({m}) must match B row count ({m2})"
    assert n == n2, f"Matrix A row count ({n}) must match C row count ({n2})"
    assert p == p2, f"Matrix B column count ({p}) must match C column count ({p2})"
    
    # Create output tensor
    out = torch.empty_like(C)
    
    # First operation: C = alpha * A @ B + beta * C
    BLOCK = 16
    grid = (triton.cdiv(n, BLOCK), triton.cdiv(p, BLOCK))
    
    # Create intermediate tensor for first operation
    intermediate = torch.empty_like(C)
    
    _matmul_kernel[grid](
        A, B, C, intermediate,
        n, m, p,
        alpha, beta,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        intermediate.stride(0), intermediate.stride(1),
        BLOCK=BLOCK
    )
    
    # Second operation: C = alpha * C @ C.T + beta * C
    # We need to compute C @ C.T where C is n x p
    # Result will be n x n
    _symmetric_update_kernel[grid](
        intermediate, out,
        n, p,
        alpha, beta,
        intermediate.stride(0), intermediate.stride(1),
        out.stride(0), out.stride(1),
        BLOCK=BLOCK
    )
    
    return out

##################################################################################################################################################



import torch

def test_matrix_multiply_symmetric():
    results = {}

    # Test Case 1: Basic test with 2x2 matrices
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[0.5, -1.0], [1.5, 2.0]], device='cuda')
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    alpha, beta = 2.0, 0.5
    results["test_case_1"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    # Test Case 2: Test with identity matrices
    A = torch.eye(3, device='cuda')
    B = torch.eye(3, device='cuda')
    C = torch.eye(3, device='cuda')
    alpha, beta = 1.0, 1.0
    results["test_case_2"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    # Test Case 3: Test with zero matrices
    A = torch.zeros((2, 2), device='cuda')
    B = torch.zeros((2, 2), device='cuda')
    C = torch.zeros((2, 2), device='cuda')
    alpha, beta = 1.0, 1.0
    results["test_case_3"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    # Test Case 4: Test with different alpha and beta
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    B = torch.tensor([[0.5, -1.0], [1.5, 2.0]], device='cuda')
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    alpha, beta = 0.5, 2.0
    results["test_case_4"] = matrix_multiply_symmetric(A, B, C, alpha, beta)

    return results

test_results = test_matrix_multiply_symmetric()
