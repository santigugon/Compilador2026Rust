import torch
import triton
import triton.language as tl

@triton.jit
def _svd_reconstruct_kernel(u_ptr, s_ptr, vh_ptr, out_ptr, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr, 
                          u_stride_0: tl.constexpr, u_stride_1: tl.constexpr, 
                          s_stride: tl.constexpr,
                          vh_stride_0: tl.constexpr, vh_stride_1: tl.constexpr,
                          out_stride_0: tl.constexpr, out_stride_1: tl.constexpr,
                          BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Each block handles one element of the output matrix
    if pid_m * BLOCK >= m or pid_n * BLOCK >= n:
        return
        
    # Load the singular values (assuming s is 1D)
    s = tl.load(s_ptr + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < k, other=0.0)
    
    # Compute the reconstruction for this block
    for i in range(BLOCK):
        for j in range(BLOCK):
            if pid_m * BLOCK + i < m and pid_n * BLOCK + j < n:
                # Compute dot product of u_row and vh_col weighted by singular values
                acc = 0.0
                for l in range(k):
                    u_val = tl.load(u_ptr + (pid_m * BLOCK + i) * u_stride_0 + l * u_stride_1, 
                                  mask=l < k, other=0.0)
                    vh_val = tl.load(vh_ptr + l * vh_stride_0 + (pid_n * BLOCK + j) * vh_stride_1, 
                                   mask=l < k, other=0.0)
                    acc += u_val * s[l] * vh_val
                tl.store(out_ptr + (pid_m * BLOCK + i) * out_stride_0 + (pid_n * BLOCK + j) * out_stride_1, 
                        acc)

def fused_svd_reconstruct(A):
    # Perform SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Reconstruct the matrix
    m, n = A.shape
    k = S.shape[0]
    
    # Create output tensor
    out = torch.empty_like(A)
    
    # Get strides
    u_stride_0, u_stride_1 = U.stride(0), U.stride(1)
    s_stride = S.stride(0) if len(S.shape) > 0 else 1
    vh_stride_0, vh_stride_1 = Vh.stride(0), Vh.stride(1)
    out_stride_0, out_stride_1 = out.stride(0), out.stride(1)
    
    # Launch kernel
    BLOCK = 16
    grid_m = triton.cdiv(m, BLOCK)
    grid_n = triton.cdiv(n, BLOCK)
    grid = (grid_m, grid_n)
    
    _svd_reconstruct_kernel[grid](
        U, S, Vh, out,
        m, n, k,
        u_stride_0, u_stride_1,
        s_stride,
        vh_stride_0, vh_stride_1,
        out_stride_0, out_stride_1,
        BLOCK=BLOCK
    )
    
    return out
