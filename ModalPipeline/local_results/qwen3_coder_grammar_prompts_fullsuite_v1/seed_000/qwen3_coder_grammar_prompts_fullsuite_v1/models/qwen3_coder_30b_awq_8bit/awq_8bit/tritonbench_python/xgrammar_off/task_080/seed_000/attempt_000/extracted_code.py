import torch
import triton
import triton.language as tl

@triton.jit
def _qr_solve_kernel(A_ptr, b_ptr, x_ptr, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr, 
                     A_stride_0: tl.constexpr, A_stride_1: tl.constexpr,
                     b_stride_0: tl.constexpr, b_stride_1: tl.constexpr,
                     x_stride_0: tl.constexpr, x_stride_1: tl.constexpr,
                     BLOCK: tl.constexpr):
    # Compute Q^T * b
    pid = tl.program_id(0)
    if pid < k:
        for i in range(n):
            acc = 0.0
            for j in range(m):
                a_val = tl.load(A_ptr + j * A_stride_0 + i * A_stride_1)
                b_val = tl.load(b_ptr + j * b_stride_0 + pid * b_stride_1)
                acc += a_val * b_val
            tl.store(x_ptr + i * x_stride_0 + pid * x_stride_1, acc)
    
    # Solve R * x = Q^T * b using back substitution
    if pid < k:
        for i in range(n - 1, -1, -1):
            acc = tl.load(x_ptr + i * x_stride_0 + pid * x_stride_1)
            for j in range(i + 1, n):
                r_val = tl.load(A_ptr + i * A_stride_0 + j * A_stride_1)
                x_val = tl.load(x_ptr + j * x_stride_0 + pid * x_stride_1)
                acc -= r_val * x_val
            r_diag = tl.load(A_ptr + i * A_stride_0 + i * A_stride_1)
            tl.store(x_ptr + i * x_stride_0 + pid * x_stride_1, acc / r_diag)

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    k = b.shape[1] if len(b.shape) > 1 else 1
    
    # Perform QR decomposition using torch's built-in function
    Q, R = torch.linalg.qr(A, mode='reduced')
    
    # Compute Q^T * b
    Qt_b = torch.matmul(Q.t(), b)
    
    # Solve R * x = Q^T * b using back substitution
    x = torch.zeros((n, k), dtype=A.dtype, device=A.device)
    
    # Use Triton kernel for solving
    block = 256
    grid = (triton.cdiv(k, block),)
    
    # Create a temporary tensor for the result
    out = torch.empty_like(Qt_b)
    
    # Copy Qt_b to out for the kernel
    out.copy_(Qt_b)
    
    # Launch kernel
    _qr_solve_kernel[grid](
        R, out, x,
        m, n, k,
        R.stride(0), R.stride(1),
        out.stride(0), out.stride(1),
        x.stride(0), x.stride(1),
        BLOCK=block
    )
    
    return x
