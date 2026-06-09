import torch
import triton
import triton.language as tl

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # First operation: C = alpha * torch.mm(A, B) + beta * C
    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    n, m = A.shape
    m, p = B.shape
    assert C.shape == (n, p), "C must have shape (n, p)"
    
    # First operation
    out1 = torch.empty(n, p, dtype=C.dtype, device=C.device)
    _gemm_kernel[1, 1](A, B, out1, alpha, beta, n, m, p, A.stride(0), A.stride(1), B.stride(0), B.stride(1), out1.stride(0), out1.stride(1))
    
    # Second operation
    out2 = torch.empty(n, p, dtype=C.dtype, device=C.device)
    _gemm_symmetric_kernel[1, 1](out1, out2, alpha, beta, n, p, out1.stride(0), out1.stride(1), out2.stride(0), out2.stride(1))
    
    return out2

@triton.jit
def _gemm_kernel(A_ptr, B_ptr, out_ptr, alpha: tl.constexpr, beta: tl.constexpr, n: tl.constexpr, m: tl.constexpr, p: tl.constexpr, stride_a0: tl.constexpr, stride_a1: tl.constexpr, stride_b0: tl.constexpr, stride_b1: tl.constexpr, stride_out0: tl.constexpr, stride_out1: tl.constexpr):
    # Compute C = alpha * torch.mm(A, B) + beta * C
    pid = tl.program_id(0)
    if pid == 0:
        for i in range(n):
            for j in range(p):
                sum = 0.0
                for k in range(m):
                    sum += tl.load(A_ptr + i * stride_a0 + k * stride_a1) * tl.load(B_ptr + k * stride_b0 + j * stride_b1)
                out_val = alpha * sum + beta * tl.load(out_ptr + i * stride_out0 + j * stride_out1)
                tl.store(out_ptr + i * stride_out0 + j * stride_out1, out_val)

@triton.jit
def _gemm_symmetric_kernel(C_ptr, out_ptr, alpha: tl.constexpr, beta: tl.constexpr, n: tl.constexpr, p: tl.constexpr, stride_c0: tl.constexpr, stride_c1: tl.constexpr, stride_out0: tl.constexpr, stride_out1: tl.constexpr):
    # Compute C = alpha * torch.mm(C, C.T) + beta * C
    pid = tl.program_id(0)
    if pid == 0:
        for i in range(n):
            for j in range(p):
                sum = 0.0
                for k in range(p):
                    sum += tl.load(C_ptr + i * stride_c0 + k * stride_c1) * tl.load(C_ptr + j * stride_c0 + k * stride_c1)
                out_val = alpha * sum + beta * tl.load(out_ptr + i * stride_out0 + j * stride_out1)
                tl.store(out_ptr + i * stride_out0 + j * stride_out1, out_val)
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
