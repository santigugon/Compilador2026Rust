import torch
import triton
import triton.language as tl

@triton.jit
def _solve_multiple_lu_kernel(A_ptr, Bs_ptr, out_ptr, n, k, batch_size, stride_A, stride_Bs, stride_out, BLOCK_SIZE, pivot: tl.constexpr):
    batch_idx = tl.program_id(0)
    if batch_idx >= batch_size:
        return
    
    A_base = A_ptr + batch_idx * stride_A
    Bs_base = Bs_ptr + batch_idx * stride_Bs
    out_base = out_ptr + batch_idx * stride_out
    
    # Allocate shared memory for L and U matrices
    L = tl.shared_ptr(tl.full((n, n), 0.0, dtype=tl.float32), (n, n))
    U = tl.shared_ptr(tl.full((n, n), 0.0, dtype=tl.float32), (n, n))
    
    # Copy A to L and U
    for i in range(n):
        for j in range(n):
            L[i, j] = tl.load(A_base + i * stride_A + j)
            U[i, j] = tl.load(A_base + i * stride_A + j)
    
    # LU decomposition with partial pivoting
    if pivot:
        for i in range(n):
            # Find pivot
            max_idx = i
            max_val = tl.abs(U[i, i])
            for j in range(i + 1, n):
                if tl.abs(U[j, i]) > max_val:
                    max_val = tl.abs(U[j, i])
                    max_idx = j
            
            # Swap rows in L and U
            if max_idx != i:
                for j in range(n):
                    temp = L[i, j]
                    L[i, j] = L[max_idx, j]
                    L[max_idx, j] = temp
                    
                    temp = U[i, j]
                    U[i, j] = U[max_idx, j]
                    U[max_idx, j] = temp
            
            # Forward elimination
            for j in range(i + 1, n):
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                for k in range(i + 1, n):
                    U[j, k] = U[j, k] - factor * U[i, k]
    
    # Solve Ly = b
    for i in range(n):
        for j in range(k):
            y = tl.load(Bs_base + i * stride_Bs + j)
            for l in range(i):
                y = y - L[i, l] * tl.load(out_base + l * stride_out + j)
            tl.store(out_base + i * stride_out + j, y)
    
    # Solve Ux = y
    for i in range(n - 1, -1, -1):
        for j in range(k):
            x = tl.load(out_base + i * stride_out + j)
            for l in range(i + 1, n):
                x = x - U[i, l] * tl.load(out_base + l * stride_out + j)
            x = x / U[i, i]
            tl.store(out_base + i * stride_out + j, x)

def solve_multiple_lu(A, Bs, *, pivot=True, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(Bs)
    
    # Validate input shapes
    assert A.dim() >= 2, "A must have at least 2 dimensions"
    assert Bs.dim() >= 2, "Bs must have at least 2 dimensions"
    assert A.shape[-1] == A.shape[-2], "A must be square"
    assert A.shape[-1] == Bs.shape[-2], "A and Bs must have compatible dimensions"
    
    # Get batch dimensions
    batch_shape = A.shape[:-2]
    n = A.shape[-1]
    k = Bs.shape[-1]
    
    # Flatten batch dimensions
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    
    # Prepare strides
    stride_A = A.stride(-2) if A.dim() > 2 else 1
    stride_Bs = Bs.stride(-2) if Bs.dim() > 2 else 1
    stride_out = out.stride(-2) if out.dim() > 2 else 1
    
    # Launch kernel
    grid = (batch_size,)
    BLOCK_SIZE = 32
    
    _solve_multiple_lu_kernel[grid](
        A.data_ptr(),
        Bs.data_ptr(),
        out.data_ptr(),
        n,
        k,
        batch_size,
        stride_A,
        stride_Bs,
        stride_out,
        BLOCK_SIZE,
        pivot
    )
    
    return out
