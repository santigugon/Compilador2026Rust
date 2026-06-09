import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_vector_norm_kernel(A_ptr, x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, beta: tl.constexpr, p: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Compute matrix-vector product y = alpha * A @ x
    row = pid
    if row < n:
        acc = 0.0
        for col in range(0, n, BLOCK):
            col_offsets = col + tl.arange(0, BLOCK)
            mask = col_offsets < n
            a_vals = tl.load(A_ptr + row * n + col_offsets, mask=mask, other=0.0)
            x_vals = tl.load(x_ptr + col_offsets, mask=mask, other=0.0)
            acc += tl.sum(a_vals * x_vals)
        y_val = alpha * acc
        # Add beta * y
        y_val += beta * tl.load(y_ptr + row, mask=True, other=0.0)
        tl.store(y_ptr + row, y_val, mask=True)
    
    # Compute norm of y
    if pid == 0:
        norm = 0.0
        for i in range(0, n, BLOCK):
            offsets = i + tl.arange(0, BLOCK)
            mask = offsets < n
            y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
            if p == 2.0:
                norm += tl.sum(y_vals * y_vals)
            else:
                norm += tl.sum(tl.abs(y_vals) ** p)
        if p == 2.0:
            norm = tl.sqrt(norm)
        else:
            norm = norm ** (1.0 / p)
        tl.store(out_ptr, norm, mask=True)


def symmetric_matrix_vector_norm(A: torch.Tensor, x: torch.Tensor, alpha: float, beta: float, p: float = 2.0) -> torch.Tensor:
    assert A.shape == (x.shape[0], x.shape[0]), "Matrix A must be square and compatible with vector x"
    n = A.shape[0]
    y = torch.empty_like(x)
    out = torch.empty((), dtype=torch.float32)
    
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _matrix_vector_norm_kernel[grid](A, x, y, out, n, alpha, beta, p, BLOCK=block)
    return out