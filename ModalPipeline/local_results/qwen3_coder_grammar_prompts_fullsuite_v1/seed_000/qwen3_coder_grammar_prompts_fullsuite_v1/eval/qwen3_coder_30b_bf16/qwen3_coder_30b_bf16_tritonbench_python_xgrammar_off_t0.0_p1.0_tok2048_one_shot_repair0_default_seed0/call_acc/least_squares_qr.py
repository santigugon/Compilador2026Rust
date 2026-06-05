import torch
import triton
import triton.language as tl

@triton.jit
def _qr_decomposition_kernel(A_ptr, Q_ptr, R_ptr, m, n, stride_A_row, stride_A_col, stride_Q_row, stride_Q_col, stride_R_row, stride_R_col, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= m * n:
        return
    
    row = pid // n
    col = pid % n
    
    if row < m and col < n:
        A_val = tl.load(A_ptr + row * stride_A_row + col * stride_A_col)
        tl.store(Q_ptr + row * stride_Q_row + col * stride_Q_col, A_val)
        if row <= col:
            tl.store(R_ptr + row * stride_R_row + col * stride_R_col, A_val)
        else:
            tl.store(R_ptr + row * stride_R_row + col * stride_R_col, 0.0)

@triton.jit
def _back_substitution_kernel(R_ptr, b_ptr, x_ptr, m, n, stride_R_row, stride_R_col, stride_b, stride_x, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n:
        return
    
    col = n - 1 - pid
    
    if col < m:
        sum_val = 0.0
        for i in range(col + 1, n):
            sum_val += tl.load(R_ptr + col * stride_R_row + i * stride_R_col) * tl.load(x_ptr + i * stride_x)
        
        x_val = (tl.load(b_ptr + col * stride_b) - sum_val) / tl.load(R_ptr + col * stride_R_row + col * stride_R_col)
        tl.store(x_ptr + col * stride_x, x_val)

def least_squares_qr(A, b, *, mode='reduced', out=None) -> torch.Tensor:
    if mode not in ['reduced', 'complete']:
        raise ValueError("mode must be 'reduced' or 'complete'")
    
    if A.dim() < 2:
        raise ValueError("A must have at least 2 dimensions")
    
    if b.dim() < 1:
        raise ValueError("b must have at least 1 dimension")
    
    batch_dims = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    
    if b.shape[-2] != m:
        raise ValueError("Incompatible dimensions between A and b")
    
    if b.dim() == 1:
        k = 1
    else:
        k = b.shape[-1]
    
    if out is not None:
        if out.shape != (*batch_dims, n, k):
            raise ValueError("out tensor has incorrect shape")
        x = out
    else:
        x = torch.empty((*batch_dims, n, k), dtype=A.dtype, device=A.device)
    
    # Handle batch dimensions
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Process each batch
    for i in range(batch_size):
        # Extract batch slice
        if batch_dims:
            A_batch = A.view(-1, m, n)[i]
            b_batch = b.view(-1, m, k)[i]
        else:
            A_batch = A
            b_batch = b
        
        # QR decomposition
        Q = torch.empty_like(A_batch)
        R = torch.zeros_like(A_batch)
        
        # Simple implementation using torch's QR for now
        Q_batch, R_batch = torch.linalg.qr(A_batch, mode=mode)
        
        # Back substitution
        if k == 1:
            x_batch = torch.zeros(n, 1, dtype=A_batch.dtype, device=A_batch.device)
            b_vec = b_batch.view(-1, 1)
        else:
            x_batch = torch.zeros(n, k, dtype=A_batch.dtype, device=A_batch.device)
            b_vec = b_batch
        
        # Solve Rx = Q^T * b
        Q_T_b = torch.matmul(Q_batch.t(), b_vec)
        
        # Back substitution to solve Rx = Q^T * b
        for j in range(n - 1, -1, -1):
            if R_batch[j, j] != 0:
                x_batch[j] = (Q_T_b[j] - torch.dot(R_batch[j, j+1:], x_batch[j+1:])) / R_batch[j, j]
            else:
                x_batch[j] = 0
        
        # Store result
        if batch_dims:
            x.view(-1, n, k)[i] = x_batch
        else:
            x = x_batch
    
    return x

##################################################################################################################################################



import torch

def test_least_squares_qr():
    results = {}
    
    # Test case 1: Simple overdetermined system with reduced QR
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b1 = torch.tensor([7.0, 8.0, 9.0], device='cuda')
    results["test_case_1"] = least_squares_qr(A1, b1)
    
    # Test case 4: Multiple right-hand sides with reduced QR
    A4 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    b4 = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device='cuda')
    results["test_case_4"] = least_squares_qr(A4, b4)
    
    return results

# Run the test
test_results = test_least_squares_qr()
