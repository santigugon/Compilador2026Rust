import torch
import triton
import triton.language as tl

@triton.jit
def _cholesky_solve_kernel(
    B_ptr, L_ptr, out_ptr,
    batch_size, n, k,
    stride_b_batch, stride_b_n, stride_b_k,
    stride_l_batch, stride_l_n, stride_l_n,
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
        
        # Compute sum for forward substitution
        sum_val = 0.0
        if upper:
            # Upper triangular: L[i, j] * Y[j] for j > i
            for j in range(i + 1, n):
                l_val = tl.load(L_batch + i * stride_l_n + j * stride_l_n)
                y_val = tl.load(out_batch + j * stride_out_n + k_idx * stride_out_k)
                sum_val += l_val * y_val
        else:
            # Lower triangular: L[i, j] * Y[j] for j < i
            for j in range(i):
                l_val = tl.load(L_batch + i * stride_l_n + j * stride_l_n)
                y_val = tl.load(out_batch + j * stride_out_n + k_idx * stride_out_k)
                sum_val += l_val * y_val
        
        # Compute Y[i]
        if upper:
            l_diag = tl.load(L_batch + i * stride_l_n + i * stride_l_n)
        else:
            l_diag = tl.load(L_batch + i * stride_l_n + i * stride_l_n)
        
        y_val = (b_val - sum_val) / l_diag
        tl.store(out_batch + i * stride_out_n + k_idx * stride_out_k, y_val)
    
    # Solve L^T * X = Y for X (backward substitution)
    for i in range(n - 1, -1, -1):
        # Load Y[i, k_idx]
        y_val = tl.load(out_batch + i * stride_out_n + k_idx * stride_out_k)
        
        # Compute sum for backward substitution
        sum_val = 0.0
        if upper:
            # Upper triangular: L^T[i, j] * X[j] for j < i
            for j in range(i):
                l_val = tl.load(L_batch + j * stride_l_n + i * stride_l_n)
                x_val = tl.load(out_batch + j * stride_out_n + k_idx * stride_out_k)
                sum_val += l_val * x_val
        else:
            # Lower triangular: L^T[i, j] * X[j] for j > i
            for j in range(i + 1, n):
                l_val = tl.load(L_batch + j * stride_l_n + i * stride_l_n)
                x_val = tl.load(out_batch + j * stride_out_n + k_idx * stride_out_k)
                sum_val += l_val * x_val
        
        # Compute X[i]
        if upper:
            l_diag = tl.load(L_batch + i * stride_l_n + i * stride_l_n)
        else:
            l_diag = tl.load(L_batch + i * stride_l_n + i * stride_l_n)
        
        x_val = (y_val - sum_val) / l_diag
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
    
    # Check if batch dimensions match
    if batch_dims_B != batch_dims_L:
        # Broadcast batch dimensions
        max_batch = max(len(batch_dims_B), len(batch_dims_L))
        batch_shape = []
        for i in range(max_batch):
            dim_B = batch_dims_B[-(i+1)] if i < len(batch_dims_B) else 1
            dim_L = batch_dims_L[-(i+1)] if i < len(batch_dims_L) else 1
            if dim_B == 1:
                batch_shape.append(dim_L)
            elif dim_L == 1:
                batch_shape.append(dim_B)
            else:
                assert dim_B == dim_L, "Batch dimensions must be broadcastable"
                batch_shape.append(dim_B)
        batch_shape.reverse()
        
        # Expand tensors to match batch dimensions
        B = B.expand(*batch_shape, B.shape[-2], B.shape[-1])
        L = L.expand(*batch_shape, L.shape[-2], L.shape[-1])
    
    batch_size = 1
    for dim in B.shape[:-2]:
        batch_size *= dim
    
    n = B.shape[-2]
    k = B.shape[-1]
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(B)
    else:
        assert out.shape == B.shape, "Output tensor must have the same shape as B"
    
    # Launch kernel
    grid = (batch_size, k)
    BLOCK_SIZE = 32
    
    # Get strides
    stride_b_batch = B.stride(-3) if B.dim() >= 3 else 0
    stride_b_n = B.stride(-2)
    stride_b_k = B.stride(-1)
    
    stride_l_batch = L.stride(-3) if L.dim() >= 3 else 0
    stride_l_n = L.stride(-2)
    stride_l_n2 = L.stride(-1)
    
    stride_out_batch = out.stride(-3) if out.dim() >= 3 else 0
    stride_out_n = out.stride(-2)
    stride_out_k = out.stride(-1)
    
    _cholesky_solve_kernel[grid](
        B, L, out,
        batch_size, n, k,
        stride_b_batch, stride_b_n, stride_b_k,
        stride_l_batch, stride_l_n, stride_l_n2,
        stride_out_batch, stride_out_n, stride_out_k,
        upper,
        BLOCK_SIZE
    )
    
    return out
