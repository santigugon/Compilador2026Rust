import torch
import triton
import triton.language as tl
import math

@triton.jit
def _qr_decomposition_kernel(A_ptr, R_ptr, Q_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    # Initialize Q as identity matrix
    pid = tl.program_id(0)
    if pid == 0:
        # Initialize R to zero
        for i in range(n):
            for j in range(n):
                if i >= j:
                    tl.store(R_ptr + i * n + j, 0.0)
        
        # Copy A to R
        for i in range(n):
            for j in range(n):
                tl.store(R_ptr + i * n + j, tl.load(A_ptr + i * n + j))
        
        # QR decomposition using Givens rotations
        for k in range(n):
            # Zero out elements below diagonal
            for j in range(k+1, n):
                # Compute Givens rotation
                r_kk = tl.load(R_ptr + k * n + k)
                r_jk = tl.load(R_ptr + j * n + k)
                
                # Handle case where r_kk is zero
                if r_kk == 0.0:
                    continue
                    
                # Compute cosine and sine
                norm = tl.sqrt(r_kk * r_kk + r_jk * r_jk)
                if norm == 0.0:
                    continue
                    
                c = r_kk / norm
                s = r_jk / norm
                
                # Apply rotation to rows k and j
                for i in range(n):
                    r_ki = tl.load(R_ptr + k * n + i)
                    r_ji = tl.load(R_ptr + j * n + i)
                    tl.store(R_ptr + k * n + i, c * r_ki + s * r_ji)
                    tl.store(R_ptr + j * n + i, -s * r_ki + c * r_ji)
                    
                    # Update Q matrix
                    q_ki = tl.load(Q_ptr + k * n + i)
                    q_ji = tl.load(Q_ptr + j * n + i)
                    tl.store(Q_ptr + k * n + i, c * q_ki + s * q_ji)
                    tl.store(Q_ptr + j * n + i, -s * q_ki + c * q_ji)

@triton.jit
def _determinant_from_r_kernel(R_ptr, det_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        # Compute determinant as product of diagonal elements
        det = 1.0
        for i in range(n):
            diag_element = tl.load(R_ptr + i * n + i)
            det *= diag_element
        tl.store(det_ptr, det)

def determinant_via_qr(A, *, mode='reduced', out=None):
    # Validate input
    if A.dim() != 2:
        raise ValueError("Input must be a 2D tensor")
    if A.size(0) != A.size(1):
        raise ValueError("Input must be a square matrix")
    
    n = A.size(0)
    
    # Create output tensor
    if out is None:
        out = torch.empty(1, dtype=torch.float64, device=A.device)
    else:
        if out.shape != (1,) or out.dtype != torch.float64:
            raise ValueError("out must be a scalar tensor with dtype=torch.float64")
    
    # For small matrices, use direct computation
    if n <= 2:
        if n == 1:
            return A[0, 0]
        elif n == 2:
            return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        else:
            raise ValueError("Invalid matrix size")
    
    # For larger matrices, use QR decomposition
    # Create identity matrix for Q
    Q = torch.eye(n, dtype=A.dtype, device=A.device)
    R = A.clone().to(A.dtype)
    
    # Perform QR decomposition using Givens rotations
    for k in range(n):
        for j in range(k+1, n):
            # Compute Givens rotation
            r_kk = R[k, k]
            r_jk = R[j, k]
            
            # Handle case where r_kk is zero
            if abs(r_kk) < 1e-12:
                continue
                
            # Compute cosine and sine
            norm = math.sqrt(r_kk * r_kk + r_jk * r_jk)
            if abs(norm) < 1e-12:
                continue
                
            c = r_kk / norm
            s = r_jk / norm
            
            # Apply rotation to rows k and j
            for i in range(n):
                r_ki = R[k, i]
                r_ji = R[j, i]
                R[k, i] = c * r_ki + s * r_ji
                R[j, i] = -s * r_ki + c * r_ji
                
                # Update Q matrix
                q_ki = Q[k, i]
                q_ji = Q[j, i]
                Q[k, i] = c * q_ki + s * q_ji
                Q[j, i] = -s * q_ki + c * q_ji
    
    # Compute determinant as product of diagonal elements of R
    det = 1.0
    for i in range(n):
        det *= R[i, i]
    
    # Store result
    out.fill_(det)
    
    return out
