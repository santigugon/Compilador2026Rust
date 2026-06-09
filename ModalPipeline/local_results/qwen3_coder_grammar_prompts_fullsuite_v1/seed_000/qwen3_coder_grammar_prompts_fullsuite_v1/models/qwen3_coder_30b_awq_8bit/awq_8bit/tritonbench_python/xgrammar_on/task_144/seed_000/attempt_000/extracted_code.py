import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, n, m, p, stride_a_row, stride_a_col, stride_b_row, stride_b_col, stride_c_row, stride_c_col, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Load A and B tiles
    a_ptrs = A_ptr + (offs_m[:, None] * stride_a_row + offs_k[None, :])
    b_ptrs = B_ptr + (offs_k[:, None] * stride_b_row + offs_n[None, :])
    
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < n) & (offs_k[None, :] < m), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < m) & (offs_n[None, :] < p), other=0.0)
    
    # Compute dot product
    c = tl.dot(a, b)
    
    # Scale and add to C
    c = alpha * c + beta * tl.load(C_ptr + (offs_m[:, None] * stride_c_row + offs_n[None, :]), mask=(offs_m[:, None] < n) & (offs_n[None, :] < p), other=0.0)
    
    # Store result
    tl.store(C_ptr + (offs_m[:, None] * stride_c_row + offs_n[None, :]), c, mask=(offs_m[:, None] < n) & (offs_n[None, :] < p))

@triton.jit
def _row_dot_kernel(C_ptr, out_ptr, n, p, stride_c_row, stride_c_col, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Compute block offsets
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load first two rows
    row0 = tl.load(C_ptr + (0 * stride_c_row + offs[None, :]), mask=(offs[None, :] < p), other=0.0)
    row1 = tl.load(C_ptr + (1 * stride_c_row + offs[None, :]), mask=(offs[None, :] < p), other=0.0)
    
    # Compute dot product of first two rows
    dot = tl.sum(row0 * row1, axis=1)
    
    # Store result
    tl.store(out_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), dot, mask=(pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < p))

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous
    A = A.contiguous()
    B = B.contiguous()
    C = C.contiguous()
    
    n, m = A.shape
    m2, p = B.shape
    
    if m != m2:
        raise ValueError(f"Matrix dimensions incompatible: {A.shape} and {B.shape}")
    
    # Create output tensor
    out = torch.empty(p, dtype=torch.float32, device=A.device)
    
    # Launch matrix multiplication kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    grid_m = triton.cdiv(n, BLOCK_SIZE_M)
    grid_n = triton.cdiv(p, BLOCK_SIZE_N)
    grid = (grid_m, grid_n)
    
    _matmul_kernel[grid](
        A, B, C, alpha, beta, n, m, p,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    # Launch row dot product kernel
    BLOCK_SIZE = 32
    grid = (triton.cdiv(p, BLOCK_SIZE),)
    
    _row_dot_kernel[grid](
        C, out, n, p,
        C.stride(0), C.stride(1),
        BLOCK_SIZE
    )
    
    return out