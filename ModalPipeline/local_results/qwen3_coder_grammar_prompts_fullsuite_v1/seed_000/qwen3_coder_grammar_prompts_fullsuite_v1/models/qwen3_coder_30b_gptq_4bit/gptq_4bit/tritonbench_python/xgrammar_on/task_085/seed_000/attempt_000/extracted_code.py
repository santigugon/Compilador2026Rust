import torch
import triton
import triton.language as tl

def matrix_power_eig(A, k, *, out=None):
    if out is None:
        out = torch.empty_like(A)
    else:
        assert out.shape == A.shape, "Output tensor must have the same shape as input tensor A"
    
    # Handle scalar k
    if isinstance(k, (int, float)):
        k = float(k)
    elif isinstance(k, complex):
        pass
    else:
        raise ValueError("k must be a real or complex number")
    
    # For batched matrices, we process each batch separately
    batch_dims = A.shape[:-2]
    n = A.shape[-1]
    
    # Flatten batch dimensions for easier processing
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Reshape A to (batch_size, n, n)
    A_flat = A.view(batch_size, n, n)
    out_flat = out.view(batch_size, n, n)
    
    # Process each matrix in the batch
    for i in range(batch_size):
        A_i = A_flat[i]
        out_i = out_flat[i]
        
        # Compute eigenvalues and eigenvectors
        try:
            # Use torch's eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eig(A_i)
            
            # Compute eigenvalues raised to power k
            eigenvals_k = torch.pow(eigenvals, k)
            
            # Reconstruct the matrix power
            # A^k = V * diag(eigenvals^k) * V^(-1)
            # Note: V^(-1) is the inverse of the eigenvector matrix
            diag_k = torch.diag(eigenvals_k)
            out_i.copy_(torch.matmul(torch.matmul(eigenvecs, diag_k), torch.linalg.inv(eigenvecs)))
        except Exception:
            # If eigendecomposition fails, fall back to a simple approach
            # This is a simplified fallback and may not be accurate for all cases
            out_i.copy_(torch.matrix_power(A_i, k))
    
    # Reshape back to original shape
    out = out_flat.view(A.shape)
    return out