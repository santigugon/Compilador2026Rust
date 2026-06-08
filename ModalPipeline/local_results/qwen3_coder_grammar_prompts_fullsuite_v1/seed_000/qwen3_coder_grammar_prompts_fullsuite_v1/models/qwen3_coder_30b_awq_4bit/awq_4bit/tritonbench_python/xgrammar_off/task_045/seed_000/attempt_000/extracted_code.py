import torch
import triton
import triton.language as tl

@triton.jit
def svd_reconstruct_kernel(
    U_ptr, S_ptr, Vh_ptr,
    output_ptr,
    m, n, k,
    stride_um, stride_un,
    stride_sm, stride_sn,
    stride_vhm, stride_vhn,
    stride_out_m, stride_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output matrix dimensions
    out_m = m
    out_n = n
    
    # Initialize accumulator for dot product
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, k, BLOCK_SIZE_K):
        # Load U block
        u_block = tl.load(
            U_ptr + pid_m * stride_um + tl.arange(0, BLOCK_SIZE_M)[:, None] * stride_um + 
            tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_un
        )
        
        # Load S block
        s_block = tl.load(
            S_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_sm + 
            tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_sn
        )
        
        # Load Vh block
        vh_block = tl.load(
            Vh_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_vhm + 
            pid_n * stride_vhn + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_vhn
        )
        
        # Compute dot product
        acc += tl.dot(u_block, s_block * vh_block)
    
    # Write output
    output_block = acc
    tl.store(
        output_ptr + pid_m * stride_out_m + pid_n * stride_out_n,
        output_block
    )

def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    # Ensure input is on GPU and in float32
    if A.device.type != 'cuda':
        A = A.cuda()
    if A.dtype != torch.float32:
        A = A.float()
    
    # Perform SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Get dimensions
    m, n = A.shape
    k = S.shape[0]
    
    # Create output tensor
    output = torch.empty(m, n, device=A.device, dtype=A.dtype)
    
    # Launch kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Define grid
    grid = (
        triton.cdiv(m, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N)
    )
    
    # Launch kernel
    svd_reconstruct_kernel[grid](
        U, S, Vh,
        output,
        m, n, k,
        U.stride(0), U.stride(1),
        S.stride(0), S.stride(1),
        Vh.stride(0), Vh.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output
