import torch
import triton
import triton.language as tl

@triton.jit
def _qr_decomp_kernel(A, Q, R, m, n, stride_A_row, stride_A_col, stride_Q_row, stride_Q_col, stride_R_row, stride_R_col, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= m * n:
        return
    
    row = pid // n
    col = pid % n
    
    if row < m and col < n:
        A_val = tl.load(A + row * stride_A_row + col * stride_A_col)
        if row == col:
            tl.store(R + row * stride_R_row + col * stride_R_col, A_val)
        elif row < col:
            tl.store(R + row * stride_R_row + col * stride_R_col, A_val)
        else:
            tl.store(Q + row * stride_Q_row + col * stride_Q_col, A_val)

@triton.jit
def _solve_triangular_kernel(R, b, x, m, n, k, stride_R_row, stride_R_col, stride_b_row, stride_b_col, stride_x_row, stride_x_col, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= k:
        return
    
    col = pid
    
    for i in range(n - 1, -1, -1):
        if i < m:
            sum_val = 0.0
            for j in range(i + 1, n):
                if j < m and col < k:
                    sum_val += tl.load(R + i * stride_R_row + j * stride_R_col) * tl.load(x + j * stride_x_row + col * stride_x_col)
            
            b_val = tl.load(b + i * stride_b_row + col * stride_b_col)
            x_val = (b_val - sum_val) / tl.load(R + i * stride_R_row + i * stride_R_col)
            tl.store(x + i * stride_x_row + col * stride_x_col, x_val)

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    k = b.shape[1]
    
    # Allocate output tensors
    Q = torch.empty_like(A)
    R = torch.zeros((n, n), device=A.device, dtype=A.dtype)
    
    # QR decomposition
    grid = (m * n + 256 - 1) // 256
    _qr_decomp_kernel[grid](
        A, Q, R, m, n, A.stride(0), A.stride(1), 
        Q.stride(0), Q.stride(1), R.stride(0), R.stride(1), 
        BLOCK_SIZE=256
    )
    
    # Compute Q^T * b
    Q_T_b = torch.matmul(Q.t(), b)
    
    # Solve Rx = Q^T * b
    x = torch.zeros((n, k), device=A.device, dtype=A.dtype)
    
    grid = (k + 256 - 1) // 256
    _solve_triangular_kernel[grid](
        R, Q_T_b, x, m, n, k, 
        R.stride(0), R.stride(1), 
        Q_T_b.stride(0), Q_T_b.stride(1), 
        x.stride(0), x.stride(1), 
        BLOCK_SIZE=256
    )
    
    return x

##################################################################################################################################################



import torch

def test_fused_qr_solve():
    results = {}

    # Test case 1: Square matrix A and vector b
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b1 = torch.tensor([[5.0], [6.0]], device='cuda')
    results["test_case_1"] = fused_qr_solve(A1, b1)

    # Test case 2: Rectangular matrix A (m > n) and vector b
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b2 = torch.tensor([[7.0], [8.0], [9.0]], device='cuda')
    results["test_case_2"] = fused_qr_solve(A2, b2)

    # Test case 3: Square matrix A and matrix b with multiple columns
    A3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = fused_qr_solve(A3, b3)

    # Test case 4: Rectangular matrix A (m > n) and matrix b with multiple columns
    A4 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b4 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device='cuda')
    results["test_case_4"] = fused_qr_solve(A4, b4)

    return results

test_results = test_fused_qr_solve()
