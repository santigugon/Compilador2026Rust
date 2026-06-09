import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, out_ptr, 
                   n: tl.constexpr, m: tl.constexpr, p: tl.constexpr,
                   alpha: tl.constexpr, beta: tl.constexpr,
                   A_stride_0: tl.constexpr, A_stride_1: tl.constexpr,
                   B_stride_0: tl.constexpr, B_stride_1: tl.constexpr,
                   C_stride_0: tl.constexpr, C_stride_1: tl.constexpr,
                   out_stride_0: tl.constexpr, out_stride_1: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    # Compute output indices
    row = pid
    col = pid2
    
    if row < n and col < p:
        # Compute dot product for C = alpha * A @ B + beta * C
        acc = 0.0
        for k in range(0, m, BLOCK_SIZE):
            # Load A row slice
            a_ptrs = A_ptr + row * A_stride_0 + k * A_stride_1
            a_mask = (k + tl.arange(0, BLOCK_SIZE)) < m
            a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
            
            # Load B column slice
            b_ptrs = B_ptr + k * B_stride_0 + col * B_stride_1
            b_mask = (k + tl.arange(0, BLOCK_SIZE)) < m
            b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            # Compute partial dot product
            acc += tl.sum(a_vals * b_vals)
        
        # Compute C = alpha * A @ B + beta * C
        c_val = tl.load(C_ptr + row * C_stride_0 + col * C_stride_1)
        out_val = alpha * acc + beta * c_val
        
        # Store result
        tl.store(out_ptr + row * out_stride_0 + col * out_stride_1, out_val)

@triton.jit
def _symmetric_update_kernel(C_ptr, out_ptr, 
                            n: tl.constexpr, p: tl.constexpr,
                            alpha: tl.constexpr, beta: tl.constexpr,
                            C_stride_0: tl.constexpr, C_stride_1: tl.constexpr,
                            out_stride_0: tl.constexpr, out_stride_1: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pid2 = tl.program_id(1)
    
    # Compute output indices
    row = pid
    col = pid2
    
    if row < n and col < p:
        # Compute dot product for C = alpha * C @ C.T + beta * C
        acc = 0.0
        for k in range(0, p, BLOCK_SIZE):
            # Load C row slice
            c1_ptrs = C_ptr + row * C_stride_0 + k * C_stride_1
            c1_mask = (k + tl.arange(0, BLOCK_SIZE)) < p
            c1_vals = tl.load(c1_ptrs, mask=c1_mask, other=0.0)
            
            # Load C.T column slice (which is C.T[row, col] = C[col, row])
            c2_ptrs = C_ptr + k * C_stride_0 + col * C_stride_1
            c2_mask = (k + tl.arange(0, BLOCK_SIZE)) < p
            c2_vals = tl.load(c2_ptrs, mask=c2_mask, other=0.0)
            
            # Compute partial dot product
            acc += tl.sum(c1_vals * c2_vals)
        
        # Compute C = alpha * C @ C.T + beta * C
        c_val = tl.load(C_ptr + row * C_stride_0 + col * C_stride_1)
        out_val = alpha * acc + beta * c_val
        
        # Store result
        tl.store(out_ptr + row * out_stride_0 + col * out_stride_1, out_val)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    # First operation: C = alpha * torch.mm(A, B) + beta * C
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "Matrix dimensions incompatible for multiplication"
    assert C.shape == (n, p), "C must have shape (n, p)"
    
    # Create intermediate tensor for first operation
    C_intermediate = torch.empty_like(C)
    
    # First matrix multiplication: C = alpha * A @ B + beta * C
    block_size = 32
    grid = (triton.cdiv(n, block_size), triton.cdiv(p, block_size))
    
    _matmul_kernel[grid](
        A, B, C, C_intermediate,
        n, m, p,
        alpha, beta,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        C_intermediate.stride(0), C_intermediate.stride(1),
        BLOCK_SIZE=block_size
    )
    
    # Second operation: C = alpha * C @ C.T + beta * C
    # We need to compute C = alpha * C @ C.T + beta * C
    # This is a symmetric update operation
    C_result = torch.empty_like(C_intermediate)
    
    _symmetric_update_kernel[grid](
        C_intermediate, C_result,
        n, p,
        alpha, beta,
        C_intermediate.stride(0), C_intermediate.stride(1),
        C_result.stride(0), C_result.stride(1),
        BLOCK_SIZE=block_size
    )
    
    return C_result
