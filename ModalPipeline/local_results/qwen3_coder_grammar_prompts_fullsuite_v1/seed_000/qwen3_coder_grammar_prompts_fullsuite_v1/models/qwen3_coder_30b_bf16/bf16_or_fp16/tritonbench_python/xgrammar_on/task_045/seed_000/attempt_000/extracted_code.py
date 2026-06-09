import torch
import triton
import triton.language as tl

@triton.jit
def _svd_reconstruct_kernel(
    U_ptr, S_ptr, Vh_ptr,
    output_ptr,
    m, n, k,
    stride_uz, stride_us, stride_vh,
    stride_out_row, stride_out_col,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    U = tl.load(U_ptr + offs_m[:, None] * stride_uz + offs_k[None, :] * stride_us)
    S = tl.load(S_ptr + offs_k)
    Vh = tl.load(Vh_ptr + offs_k[:, None] * stride_vh + offs_n[None, :] * stride_vh)
    
    # Compute U * S
    U_S = U * S[None, :]
    
    # Compute U * S * V^H
    output = tl.dot(U_S, Vh)
    
    # Store result
    tl.store(output_ptr + offs_m[:, None] * stride_out_row + offs_n[None, :] * stride_out_col, output)


def fused_svd_reconstruct(A):
    # Assume A is of shape (m, n)
    m, n = A.shape
    
    # For SVD reconstruction, we need U, S, Vh
    # This is a simplified version - in practice, you'd compute SVD first
    # Here we assume U, S, Vh are already computed and passed
    # For demonstration, we'll compute SVD using torch
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Create output tensor
    output = torch.empty_like(A)
    
    # Launch kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    grid = (
        triton.cdiv(m, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N)
    )
    
    _svd_reconstruct_kernel[grid](
        U, S, Vh,
        output,
        m, n, min(m, n),
        U.stride(0), U.stride(1),
        Vh.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output