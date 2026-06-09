import torch
import triton
import triton.language as tl

@triton.jit
def _solve_multiple_lu_kernel(A_ptr, Bs_ptr, X_ptr, batch_shape, n, k, pivot: tl.constexpr, BLOCK: tl.constexpr):
    # Get batch indices
    batch_idx = tl.program_id(0)
    batch_size = tl.prod(batch_shape)
    
    if batch_idx >= batch_size:
        return
    
    # Compute linear batch index
    linear_batch_idx = batch_idx
    batch_indices = []
    temp = linear_batch_idx
    for i in range(len(batch_shape) - 1, -1, -1):
        batch_indices.append(temp % batch_shape[i])
        temp //= batch_shape[i]
    batch_indices.reverse()
    
    # Compute pointers for this batch
    batch_offset_A = batch_idx * n * n
    batch_offset_Bs = batch_idx * n * k
    batch_offset_X = batch_idx * n * k
    
    # Load A and Bs for this batch
    A_block = A_ptr + batch_offset_A
    Bs_block = Bs_ptr + batch_offset_Bs
    X_block = X_ptr + batch_offset_X
    
    # Create temporary arrays for LU decomposition
    L = tl.full((n, n), 0.0, dtype=tl.float32)
    U = tl.full((n, n), 0.0, dtype=tl.float32)
    P = tl.full((n,), 0, dtype=tl.int32)
    
    # Copy A to U
    for i in range(n):
        for j in range(n):
            if i == j:
                U[i, j] = tl.load(A_block + i * n + j)
            else:
                U[i, j] = tl.load(A_block + i * n + j)
    
    # Initialize P (permutation array)
    for i in range(n):
        P[i] = i
    
    # LU decomposition with partial pivoting
    if pivot:
        for i in range(n):
            # Find pivot
            max_val = tl.abs(U[i, i])
            max_row = i
            for j in range(i + 1, n):
                if tl.abs(U[j, i]) > max_val:
                    max_val = tl.abs(U[j, i])
                    max_row = j
            
            # Swap rows in U
            if max_row != i:
                for j in range(n):
                    temp = U[i, j]
                    U[i, j] = U[max_row, j]
                    U[max_row, j] = temp
                
                # Update permutation
                temp = P[i]
                P[i] = P[max_row]
                P[max_row] = temp
            
            # Eliminate
            for j in range(i + 1, n):
                if tl.abs(U[i, i]) > 1e-12:  # Avoid division by zero
                    factor = U[j, i] / U[i, i]
                    for k_idx in range(i, n):
                        U[j, k_idx] = U[j, k_idx] - factor * U[i, k_idx]
                    # Store L factor
                    L[j, i] = factor
                else:
                    L[j, i] = 0.0
    
    # Set diagonal of L to 1
    for i in range(n):
        L[i, i] = 1.0
    
    # Forward substitution: L * Y = P * B
    Y = tl.full((n, k), 0.0, dtype=tl.float32)
    
    # Permute B according to P
    for i in range(n):
        for j in range(k):
            Y[i, j] = tl.load(Bs_block + P[i] * k + j)
    
    # Forward substitution
    for i in range(n):
        for j in range(k):
            for p in range(i):
                Y[i, j] = Y[i, j] - L[i, p] * Y[p, j]
    
    # Backward substitution: U * X = Y
    for i in range(n - 1, -1, -1):
        for j in range(k):
            for p in range(i + 1, n):
                Y[i, j] = Y[i, j] - U[i, p] * Y[p, j]
            if tl.abs(U[i, i]) > 1e-12:
                Y[i, j] = Y[i, j] / U[i, i]
            else:
                Y[i, j] = 0.0
    
    # Store result
    for i in range(n):
        for j in range(k):
            tl.store(X_block + i * k + j, Y[i, j])

def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
    # Validate inputs
    assert A.dim() >= 2, "A must have at least 2 dimensions"
    assert Bs.dim() >= 2, "Bs must have at least 2 dimensions"
    assert A.shape[-1] == A.shape[-2], "A must be square"
    assert A.shape[-1] == Bs.shape[-2], "A and Bs must have compatible dimensions"
    
    # Get batch dimensions
    batch_shape_A = A.shape[:-2]
    batch_shape_Bs = Bs.shape[:-2]
    
    # Check if batch shapes are compatible
    if batch_shape_A != batch_shape_Bs:
        # Broadcast batch dimensions
        batch_shape = []
        max_dims = max(len(batch_shape_A), len(batch_shape_Bs))
        for i in range(max_dims):
            dim_A = batch_shape_A[-(i+1)] if i < len(batch_shape_A) else 1
            dim_Bs = batch_shape_Bs[-(i+1)] if i < len(batch_shape_Bs) else 1
            if dim_A == 1:
                batch_shape.append(dim_Bs)
            elif dim_Bs == 1:
                batch_shape.append(dim_A)
            else:
                assert dim_A == dim_Bs, "Batch dimensions must be broadcastable"
                batch_shape.append(dim_A)
        batch_shape.reverse()
    else:
        batch_shape = list(batch_shape_A)
    
    n = A.shape[-1]
    k = Bs.shape[-1]
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(Bs)
    else:
        assert out.shape == Bs.shape, "Output tensor must have the same shape as Bs"
    
    # Handle scalar case
    if len(batch_shape) == 0:
        batch_shape = [1]
    
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    
    # Launch kernel
    block = 16
    grid = (batch_size,)
    
    # Create a contiguous version of A and Bs for kernel access
    A_contiguous = A.contiguous()
    Bs_contiguous = Bs.contiguous()
    
    # Launch kernel
    _solve_multiple_lu_kernel[grid](
        A_contiguous, 
        Bs_contiguous, 
        out, 
        batch_shape, 
        n, 
        k, 
        pivot, 
        BLOCK=block
    )
    
    return out
