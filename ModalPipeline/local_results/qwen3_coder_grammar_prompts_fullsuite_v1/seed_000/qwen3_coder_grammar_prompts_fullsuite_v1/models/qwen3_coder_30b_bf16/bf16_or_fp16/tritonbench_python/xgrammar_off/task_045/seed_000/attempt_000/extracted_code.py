import torch
import triton
import triton.language as tl

@triton.jit
def svd_reconstruct_kernel(
    U_ptr, S_ptr, Vh_ptr,
    output_ptr,
    m, n, k,
    stride_um, stride_un,
    stride_sm,
    stride_vhm, stride_vhn,
    stride_out_m, stride_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    if m_start >= m or n_start >= n:
        return
    
    # Compute U * S
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_start in range(0, k, BLOCK_SIZE_K):
        k_limit = min(k_start + BLOCK_SIZE_K, k)
        
        # Load U
        u_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < m) & (tl.arange(0, BLOCK_SIZE_K)[None, :] < k)
        u = tl.load(U_ptr + m_start * stride_um + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_un, mask=u_mask)
        
        # Load S
        s_mask = (tl.arange(0, BLOCK_SIZE_K)[:, None] < k) & (tl.arange(0, 1)[None, :] < 1)
        s = tl.load(S_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_sm, mask=s_mask)
        
        # Compute U * S
        u_s = u @ s
        acc += u_s
    
    # Compute (U * S) * V^H
    for k_start in range(0, k, BLOCK_SIZE_K):
        k_limit = min(k_start + BLOCK_SIZE_K, k)
        
        # Load V^H
        v_mask = (tl.arange(0, BLOCK_SIZE_K)[:, None] < k) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < n)
        v = tl.load(Vh_ptr + k_start * stride_vhm + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_vhn, mask=v_mask)
        
        # Compute (U * S) * V^H
        acc = acc @ v
    
    # Store result
    out_mask = (tl.arange(0, BLOCK_SIZE_M)[:, None] < m) & (tl.arange(0, BLOCK_SIZE_N)[None, :] < n)
    tl.store(output_ptr + m_start * stride_out_m + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_out_n, acc, mask=out_mask)

def fused_svd_reconstruct(A):
    """
    Reconstructs the input matrix A using its SVD.
    
    Args:
        A: Input matrix of shape (m, n)
        
    Returns:
        Reconstructed matrix of shape (m, n)
    """
    if not torch.is_tensor(A):
        raise TypeError("Input must be a PyTorch tensor")
    
    if A.dim() != 2:
        raise ValueError("Input must be a 2D tensor")
    
    m, n = A.shape
    
    # Perform SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Ensure we have the right shapes for computation
    k = min(m, n)
    
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
    
    # Get strides for memory access
    stride_um, stride_un = U.stride()
    stride_sm = S.stride(0)
    stride_vhm, stride_vhn = Vh.stride()
    stride_out_m, stride_out_n = output.stride()
    
    # Launch kernel
    svd_reconstruct_kernel[grid](
        U, S, Vh,
        output,
        m, n, k,
        stride_um, stride_un,
        stride_sm,
        stride_vhm, stride_vhn,
        stride_out_m, stride_out_n,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output
