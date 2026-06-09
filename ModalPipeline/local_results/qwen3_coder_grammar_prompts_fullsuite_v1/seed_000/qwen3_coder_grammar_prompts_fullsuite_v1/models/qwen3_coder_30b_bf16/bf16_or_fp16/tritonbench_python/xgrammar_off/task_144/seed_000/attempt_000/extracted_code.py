import torch
import triton
import triton.language as tl

@triton.jit
def _matrix_multiply_and_row_dot_kernel(
    A_ptr, B_ptr, C_ptr,
    n, m, p,
    alpha, beta,
    stride_a_row, stride_a_col,
    stride_b_row, stride_b_col,
    stride_c_row, stride_c_col,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, m, BLOCK_SIZE_K):
        a = tl.load(A_ptr + offs_m[:, None] * stride_a_row + offs_k[None, :] * stride_a_col)
        b = tl.load(B_ptr + offs_k[:, None] * stride_b_row + offs_n[None, :] * stride_b_col)
        acc += tl.dot(a, b)
    
    acc *= alpha
    
    c = tl.load(C_ptr + offs_m[:, None] * stride_c_row + offs_n[None, :] * stride_c_col)
    acc += c * beta
    
    # Store result back to C
    tl.store(C_ptr + offs_m[:, None] * stride_c_row + offs_n[None, :] * stride_c_col, acc)
    
    # Compute dot product of first two rows
    if pid_m == 0 and pid_n == 0:
        row0 = tl.load(C_ptr + 0 * stride_c_row + offs_n[None, :] * stride_c_col)
        row1 = tl.load(C_ptr + 1 * stride_c_row + offs_n[None, :] * stride_c_col)
        dot_product = tl.sum(row0 * row1)
        # Store dot product in first element of C
        tl.store(C_ptr + 0 * stride_c_row + 0 * stride_c_col, dot_product)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    assert A.shape[1] == B.shape[0], "Matrix dimensions incompatible for multiplication"
    assert A.shape[0] == C.shape[0] and B.shape[1] == C.shape[1], "Matrix dimensions incompatible for output"
    
    n, m = A.shape
    m_, p = B.shape
    
    # Ensure tensors are contiguous and on the same device
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    # Create a copy of C to avoid modifying the original
    C_out = C.clone()
    
    # Launch kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    grid = (
        triton.cdiv(n, BLOCK_SIZE_M),
        triton.cdiv(p, BLOCK_SIZE_N)
    )
    
    _matrix_multiply_and_row_dot_kernel[grid](
        A, B, C_out,
        n, m, p,
        alpha, beta,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C_out.stride(0), C_out.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return C_out
