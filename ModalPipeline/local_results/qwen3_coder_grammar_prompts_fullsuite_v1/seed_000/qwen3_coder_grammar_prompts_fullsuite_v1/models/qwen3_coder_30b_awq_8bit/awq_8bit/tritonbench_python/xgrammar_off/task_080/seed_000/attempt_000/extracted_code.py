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
    for i in range(k):
        for j in range(n):
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for l in range(m):
                a_val = tl.load(A_ptr + l * A_stride_0 + j * A_stride_1)
                b_val = tl.load(b_ptr + l * b_stride_0 + i * b_stride_1)
                acc += a_val * b_val
            # Store intermediate result in x
            tl.store(x_ptr + j * x_stride_0 + i * x_stride_1, acc)
    
    # Solve Rx = Q^T * b using back substitution
    for i in range(k):
        for j in range(n - 1, -1, -1):
            # Load the value from x
            x_val = tl.load(x_ptr + j * x_stride_0 + i * x_stride_1)
            # Back substitution
            for l in range(j + 1, n):
                r_val = tl.load(A_ptr + j * A_stride_0 + l * A_stride_1)
                x_val -= r_val * tl.load(x_ptr + l * x_stride_0 + i * x_stride_1)
            # Divide by diagonal element
            r_diag = tl.load(A_ptr + j * A_stride_0 + j * A_stride_1)
            x_val = x_val / r_diag
            # Store result
            tl.store(x_ptr + j * x_stride_0 + i * x_stride_1, x_val)

def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = A.shape
    k = b.shape[1]
    
    # Create output tensor
    x = torch.empty(n, k, dtype=A.dtype, device=A.device)
    
    # Ensure tensors are contiguous for easier indexing
    A_contiguous = A.contiguous()
    b_contiguous = b.contiguous()
    x_contiguous = x.contiguous()
    
    # Launch kernel
    block = 16
    grid = (triton.cdiv(n, block), triton.cdiv(k, block))
    
    _qr_solve_kernel[grid](
        A_contiguous, b_contiguous, x_contiguous,
        m, n, k,
        A_contiguous.stride(0), A_contiguous.stride(1),
        b_contiguous.stride(0), b_contiguous.stride(1),
        x_contiguous.stride(0), x_contiguous.stride(1),
        BLOCK=block
    )
    
    return x_contiguous
