import torch
import triton
import triton.language as tl

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    k = b.shape[1]
    
    # Allocate output tensor
    x = torch.empty(n, k, device=A.device, dtype=A.dtype)
    
    # Check if m >= n
    if m < n:
        raise ValueError("Matrix A must have more rows than columns (m >= n)")
    
    # Allocate temporary tensors for QR decomposition
    Q = torch.empty(m, m, device=A.device, dtype=A.dtype)
    R = torch.empty(n, n, device=A.device, dtype=A.dtype)
    
    # Copy A to Q for QR decomposition
    Q.copy_(A)
    
    # Perform QR decomposition using Householder reflections
    # This is a simplified version - in practice, you'd use a more robust implementation
    # For this example, we'll use a basic Householder approach
    
    # Initialize R with zeros
    R.zero_()
    
    # Simple Householder QR decomposition (simplified for demonstration)
    # In a real implementation, this would be more complex
    for i in range(n):
        # Compute the Householder vector
        x = Q[i:, i]
        norm_x = torch.norm(x)
        if norm_x == 0:
            continue
        
        # Create Householder vector
        v = x.clone()
        v[0] += torch.sign(x[0]) * norm_x
        v = v / torch.norm(v)
        
        # Apply Householder reflection to Q
        Q[i:, i:] -= 2 * torch.ger(v, torch.mv(Q[i:, i:].t(), v))
        
        # Store R values
        if i < n:
            R[i, i:] = Q[i, i:]
    
    # Compute Q^T * b
    Qt_b = torch.zeros(n, k, device=A.device, dtype=A.dtype)
    Qt_b = torch.mm(Q[:n, :].t(), b)
    
    # Solve Rx = Qt_b using back substitution
    # This is a simplified version - in practice, you'd use a more robust solver
    x = torch.zeros(n, k, device=A.device, dtype=A.dtype)
    
    # Back substitution
    for i in range(n-1, -1, -1):
        x[i, :] = Qt_b[i, :]
        for j in range(i+1, n):
            x[i, :] -= R[i, j] * x[j, :]
        x[i, :] /= R[i, i]
    
    return x