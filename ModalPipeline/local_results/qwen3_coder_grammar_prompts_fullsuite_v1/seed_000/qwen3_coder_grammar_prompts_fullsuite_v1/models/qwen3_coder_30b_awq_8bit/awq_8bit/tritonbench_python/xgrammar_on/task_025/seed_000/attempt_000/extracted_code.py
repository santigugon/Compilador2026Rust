import torch
import triton
import triton.language as tl

def _qr_decomposition_kernel(A_ptr, R_ptr, Q_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    # Initialize R and Q
    for i in range(n):
        for j in range(n):
            if i <= j:
                tl.store(R_ptr + i * n + j, tl.load(A_ptr + i * n + j))
            else:
                tl.store(R_ptr + i * n + j, 0.0)
    
    # Initialize Q as identity matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                tl.store(Q_ptr + i * n + j, 1.0)
            else:
                tl.store(Q_ptr + i * n + j, 0.0)
    
    # Givens rotations
    for k in range(n):
        for i in range(k + 1, n):
            # Compute Givens rotation
            a = tl.load(R_ptr + k * n + k)
            b = tl.load(R_ptr + i * n + k)
            r = tl.sqrt(a * a + b * b)
            if r == 0.0:
                continue
            c = a / r
            s = -b / r
            
            # Apply rotation to R
            for j in range(k, n):
                temp1 = tl.load(R_ptr + k * n + j)
                temp2 = tl.load(R_ptr + i * n + j)
                tl.store(R_ptr + k * n + j, c * temp1 - s * temp2)
                tl.store(R_ptr + i * n + j, s * temp1 + c * temp2)
            
            # Apply rotation to Q
            for j in range(n):
                temp1 = tl.load(Q_ptr + k * n + j)
                temp2 = tl.load(Q_ptr + i * n + j)
                tl.store(Q_ptr + k * n + j, c * temp1 - s * temp2)
                tl.store(Q_ptr + i * n + j, s * temp1 + c * temp2)

def _determinant_from_r_kernel(R_ptr, det_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    # Compute determinant as product of diagonal elements
    det = 1.0
    for i in range(n):
        diag = tl.load(R_ptr + i * n + i)
        det *= diag
    tl.store(det_ptr, det)


def determinant_via_qr(A, *, mode='reduced', out=None):
    if mode != 'reduced':
        raise NotImplementedError("Only 'reduced' mode is supported")
    
    if A.dim() != 2 or A.size(0) != A.size(1):
        raise ValueError("Input must be a square matrix")
    
    n = A.size(0)
    if n == 0:
        return torch.tensor(1.0, dtype=A.dtype, device=A.device)
    
    # Allocate output tensor
    if out is not None:
        if out.shape != () or out.dtype != A.dtype or out.device != A.device:
            raise ValueError("Output tensor has incorrect shape, dtype, or device")
        det = out
    else:
        det = torch.empty((), dtype=A.dtype, device=A.device)
    
    # Create temporary tensors for Q and R
    R = torch.empty_like(A)
    Q = torch.empty_like(A)
    
    # Copy input to R
    R.copy_(A)
    
    # Perform QR decomposition
    block = 256
    grid = (triton.cdiv(n, block),)
    _qr_decomposition_kernel[grid](R, R, Q, n, BLOCK=block)
    
    # Compute determinant from diagonal elements of R
    _determinant_from_r_kernel[grid](R, det, n, BLOCK=block)
    
    return det