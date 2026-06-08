import torch
import triton
import triton.language as tl

@triton.jit
def _qr_decomposition_kernel(A, Q, R, m, n, BLOCK_SIZE=32):
    pid = tl.program_id(0)
    if pid >= m * n:
        return
    
    row = pid // n
    col = pid % n
    
    if row < m and col < n:
        A_val = tl.load(A + row * n + col)
        tl.store(Q + row * n + col, A_val)
        if row <= col:
            tl.store(R + row * n + col, A_val)
        else:
            tl.store(R + row * n + col, 0.0)

@triton.jit
def _solve_triangular_kernel(R, b, x, m, n, k, BLOCK_SIZE=32):
    pid = tl.program_id(0)
    if pid >= m * k:
        return
    
    row = pid // k
    col = pid % k
    
    if row < m and col < k:
        x_val = tl.load(b + row * k + col)
        for i in range(row, 0, -1):
            if i <= n:
                x_val -= tl.load(R + (i-1) * n + row) * tl.load(x + (i-1) * k + col)
        tl.store(x + row * k + col, x_val)

def least_squares_qr(A, b, *, mode='reduced', out=None) -> torch.Tensor:
    if mode not in ['reduced', 'complete']:
        raise ValueError("mode must be 'reduced' or 'complete'")
    
    # Ensure inputs are tensors
    A = torch.as_tensor(A)
    b = torch.as_tensor(b)
    
    # Handle batch dimensions
    batch_shape = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    k = b.shape[-1] if b.dim() > 1 else 1
    
    # Flatten batch dimensions for processing
    batch_size = torch.prod(torch.tensor(batch_shape)) if batch_shape else 1
    A_flat = A.view(-1, m, n)
    b_flat = b.view(-1, m, k)
    
    # Initialize output
    if out is None:
        out = torch.empty(batch_size, n, k, dtype=A.dtype, device=A.device)
    else:
        out = out.view(-1, n, k)
    
    # Process each batch
    for i in range(batch_size):
        A_batch = A_flat[i]
        b_batch = b_flat[i]
        
        # QR decomposition using Triton
        Q = torch.empty_like(A_batch)
        R = torch.zeros_like(A_batch)
        
        # Launch kernel for QR decomposition
        grid = (m * n + 31) // 32
        _qr_decomposition_kernel[grid](A_batch, Q, R, m, n)
        
        # Solve triangular system using Triton
        x = torch.empty_like(b_batch)
        grid = (m * k + 31) // 32
        _solve_triangular_kernel[grid](R, b_batch, x, m, n, k)
        
        # Store result
        out[i] = x
    
    # Reshape output to match batch dimensions
    if batch_shape:
        out = out.view(batch_shape + (n, k))
    else:
        out = out.squeeze()
    
    return out
