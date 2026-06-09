import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_and_row_dot_kernel(
    A_ptr, B_ptr, C_ptr,
    out_ptr,
    n: tl.constexpr, m: tl.constexpr, p: tl.constexpr,
    alpha: tl.constexpr, beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the matrix multiplication A @ B
    pid = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Each program handles one block of the output matrix
    offs_i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_j = pid_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load A and B tiles
    A_tile = tl.load(A_ptr + offs_i[:, None] * m + tl.arange(0, m)[None, :])
    B_tile = tl.load(B_ptr + tl.arange(0, m)[None, :] + offs_j[None, :] * m)
    
    # Compute partial dot product
    acc = tl.dot(A_tile, B_tile)
    
    # Scale by alpha
    acc = acc * alpha
    
    # Load C and add scaled C
    C_tile = tl.load(C_ptr + offs_i[:, None] * p + offs_j[None, :])
    acc = acc + beta * C_tile
    
    # Store result
    tl.store(out_ptr + offs_i[:, None] * p + offs_j[None, :], acc)

@triton.jit
def _row_dot_kernel(C_ptr, out_ptr, n: tl.constexpr, p: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute dot product of first two rows
    pid = tl.program_id(0)
    
    # Load first two rows
    row0 = tl.load(C_ptr + tl.arange(0, p))
    row1 = tl.load(C_ptr + p + tl.arange(0, p))
    
    # Compute dot product
    dot_product = tl.sum(row0 * row1)
    
    # Store result
    tl.store(out_ptr + pid, dot_product)


def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    # Get dimensions
    n, m = A.shape
    m2, p = B.shape
    
    # Check dimensions match
    if m != m2:
        raise ValueError("Matrix dimensions do not match for multiplication")
    
    # Create output tensor
    out = torch.empty_like(C)
    
    # First compute A @ B * alpha + C * beta
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE), triton.cdiv(p, BLOCK_SIZE))
    
    _matrix_multiply_and_row_dot_kernel[grid](
        A, B, C, out,
        n, m, p,
        alpha, beta,
        BLOCK_SIZE
    )
    
    # Compute dot product of first two rows
    result = torch.empty(1, dtype=torch.float32)
    grid = (1,)
    _row_dot_kernel[grid](out, result, n, p, BLOCK_SIZE)
    
    return result