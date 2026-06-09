import torch
import triton
import triton.language as tl

@triton.jit
def _cholesky_solve_kernel(
    B_ptr, L_ptr, out_ptr,
    batch_size, n, k,
    stride_b_batch, stride_b_n, stride_b_k,
    stride_l_batch, stride_l_n, stride_l_k,
    stride_out_batch, stride_out_n, stride_out_k,
    upper: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    k_idx = tl.program_id(1)
    
    if batch_idx >= batch_size:
        return
    
    # Load B and L for this batch
    B_batch = B_ptr + batch_idx * stride_b_batch
    L_batch = L_ptr + batch_idx * stride_l_batch
    out_batch = out_ptr + batch_idx * stride_out_batch
    
    # Solve L * Y = B for Y (forward substitution)
    for i in range(n):
        # Load B[i, k_idx]
        b_val = tl.load(B_batch + i * stride_b_n + k_idx * stride_b_k)
        
        # Subtract sum of L[i, j] * Y[j, k_idx] for j < i
        for j in range(i):
            if upper:
                l_val = tl.load(L_batch + j * stride_l_n + i * stride_l_k)
            else:
                l_val = tl.load(L_batch + i * stride_l_n + j * stride_l_k)
            b_val = b_val - l_val * tl.load(out_batch + j * stride_out_n + k_idx * stride_out_k)
        
        # Divide by diagonal element
        if upper:
            diag = tl.load(L_batch + i * stride_l_n + i * stride_l_k)
        else:
            diag = tl.load(L_batch + i * stride_l_n + i * stride_l_k)
        y_val = b_val / diag
        
        # Store Y[i, k_idx]
        tl.store(out_batch + i * stride_out_n + k_idx * stride_out_k, y_val)
    
    # Solve L^T * X = Y for X (backward substitution)
    for i in range(n - 1, -1, -1):
        # Load Y[i, k_idx]
        y_val = tl.load(out_batch + i * stride_out_n + k_idx * stride_out_k)
        
        # Subtract sum of L[j, i] * X[j, k_idx] for j > i
        for j in range(i + 1, n):
            if upper:
                l_val = tl.load(L_batch + i * stride_l_n + j * stride_l_k)
            else:
                l_val = tl.load(L_batch + j * stride_l_n + i * stride_l_k)
            y_val = y_val - l_val * tl.load(out_batch + j * stride_out_n + k_idx * stride_out_k)
        
        # Divide by diagonal element
        if upper:
            diag = tl.load(L_batch + i * stride_l_n + i * stride_l_k)
        else:
            diag = tl.load(L_batch + i * stride_l_n + i * stride_l_k)
        x_val = y_val / diag
        
        # Store X[i, k_idx]
        tl.store(out_batch + i * stride_out_n + k_idx * stride_out_k, x_val)

def cholesky_solve(B, L, upper=False, *, out=None):
    # Validate inputs
    assert B.dim() >= 2, "B must have at least 2 dimensions"
    assert L.dim() >= 2, "L must have at least 2 dimensions"
    assert B.shape[-2] == L.shape[-2], "Last two dimensions of B and L must match"
    assert L.shape[-1] == L.shape[-2], "L must be square"
    
    # Handle batch dimensions
    batch_dims_B = B.shape[:-2]
    batch_dims_L = L.shape[:-2]
    assert batch_dims_B == batch_dims_L, "Batch dimensions of B and L must match"
    
    batch_size = 1
    for dim in batch_dims_B:
        batch_size *= dim
    
    n = L.shape[-1]
    k = B.shape[-1]
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(B)
    else:
        assert out.shape == B.shape, "Output tensor must have the same shape as B"
    
    # Launch kernel
    grid = (batch_size, k)
    BLOCK_SIZE = 32
    
    # Determine appropriate data type for kernel
    if B.dtype == torch.float32:
        dtype = tl.float32
    elif B.dtype == torch.float64:
        dtype = tl.float64
    elif B.dtype == torch.complex64:
        dtype = tl.complex64
    elif B.dtype == torch.complex128:
        dtype = tl.complex128
    else:
        raise ValueError(f"Unsupported dtype: {B.dtype}")
    
    # Launch kernel
    _cholesky_solve_kernel[grid](
        B, L, out,
        batch_size, n, k,
        B.stride(0) if B.dim() > 2 else 1,
        B.stride(-2) if B.dim() > 2 else 1,
        B.stride(-1) if B.dim() > 2 else 1,
        L.stride(0) if L.dim() > 2 else 1,
        L.stride(-2) if L.dim() > 2 else 1,
        L.stride(-1) if L.dim() > 2 else 1,
        out.stride(0) if out.dim() > 2 else 1,
        out.stride(-2) if out.dim() > 2 else 1,
        out.stride(-1) if out.dim() > 2 else 1,
        upper,
        BLOCK_SIZE
    )
    
    return out

##################################################################################################################################################



import torch

def test_cholesky_solve():
    results = {}

    # Test case 1: Lower triangular matrix
    B1 = torch.tensor([[1.0], [2.0]], device='cuda')
    L1 = torch.tensor([[2.0, 0.0], [1.0, 1.0]], device='cuda')
    results["test_case_1"] = cholesky_solve(B1, L1)

    # Test case 2: Upper triangular matrix
    B2 = torch.tensor([[1.0], [2.0]], device='cuda')
    L2 = torch.tensor([[2.0, 1.0], [0.0, 1.0]], device='cuda')
    results["test_case_2"] = cholesky_solve(B2, L2, upper=True)

    # Test case 3: Batch of matrices, lower triangular
    B3 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device='cuda')
    L3 = torch.tensor([[[2.0, 0.0], [1.0, 1.0]], [[3.0, 0.0], [1.0, 2.0]]], device='cuda')
    results["test_case_3"] = cholesky_solve(B3, L3)

    # Test case 4: Batch of matrices, upper triangular
    B4 = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], device='cuda')
    L4 = torch.tensor([[[2.0, 1.0], [0.0, 1.0]], [[3.0, 1.0], [0.0, 2.0]]], device='cuda')
    results["test_case_4"] = cholesky_solve(B4, L4, upper=True)

    return results

test_results = test_cholesky_solve()
