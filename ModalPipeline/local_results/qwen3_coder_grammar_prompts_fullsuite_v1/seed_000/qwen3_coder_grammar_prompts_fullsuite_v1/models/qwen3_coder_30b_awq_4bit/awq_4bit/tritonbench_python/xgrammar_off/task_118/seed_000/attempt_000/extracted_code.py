import torch
import triton
import triton.language as tl

@triton.jit
def triangular_solve_kernel(A_ptr, b_ptr, y_ptr, output_ptr, n, k, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        # Solve triangular system Ax = b
        for i in range(n):
            sum_val = 0.0
            for j in range(i):
                sum_val += tl.load(A_ptr + i * n + j) * tl.load(output_ptr + j)
            tl.store(output_ptr + i, (tl.load(b_ptr + i) - sum_val) / tl.load(A_ptr + i * n + i))
    
    if pid == 1:
        # Add scaled vector alpha * y to solution x
        for i in range(n):
            tl.store(output_ptr + i, tl.load(output_ptr + i) + alpha * tl.load(y_ptr + i))

def solve_and_add_scaled_vector(A: torch.Tensor, b: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    n = A.shape[0]
    k = b.shape[1] if len(b.shape) > 1 else 1
    
    # Ensure inputs are on the same device and have the correct dtype
    device = A.device
    A = A.to(torch.float32)
    b = b.to(torch.float32)
    y = y.to(torch.float32)
    
    # Initialize output tensor
    output = torch.zeros_like(b, device=device, dtype=torch.float32)
    
    # Launch kernel
    grid = (2, 1, 1)
    triangular_solve_kernel[grid](A, b, y, output, n, k, alpha, BLOCK_SIZE=32)
    
    return output
