import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, p: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    # Compute first operation: C = alpha * torch.mm(A, B) + beta * C
    if pid2 == 0:
        # Compute C = alpha * A @ B
        acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        for k in range(0, m, BLOCK):
            a = tl.load(A_ptr + tl.arange(0, BLOCK)[:, None] * m + tl.arange(0, BLOCK)[None, :] + k)
            b = tl.load(B_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :] + k * p)
            acc += tl.dot(a, b)
        
        # Store result back to C
        c = tl.load(C_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :])
        result = alpha * acc + beta * c
        tl.store(out_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :], result)
    
    # Compute second operation: C = alpha * torch.mm(C, C.T) + beta * C
    elif pid2 == 1:
        # Compute C = alpha * C @ C.T
        acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        for k in range(0, p, BLOCK):
            c1 = tl.load(C_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :] + k)
            c2 = tl.load(C_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :] + k * p)
            acc += tl.dot(c1, c2)
        
        # Store result back to C
        c = tl.load(C_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :])
        result = alpha * acc + beta * c
        tl.store(out_ptr + tl.arange(0, BLOCK)[:, None] * p + tl.arange(0, BLOCK)[None, :], result)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # First operation: C = alpha * torch.mm(A, B) + beta * C
    C1 = alpha * torch.mm(A, B) + beta * C
    
    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    C2 = alpha * torch.mm(C1, C1.T) + beta * C1
    
    return C2