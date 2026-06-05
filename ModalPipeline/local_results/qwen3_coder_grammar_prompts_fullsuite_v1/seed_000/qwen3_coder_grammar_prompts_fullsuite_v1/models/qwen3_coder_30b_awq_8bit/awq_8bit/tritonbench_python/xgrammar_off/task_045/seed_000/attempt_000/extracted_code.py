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
    row = pid_m * BLOCK + tl.arange(0, BLOCK)
    col = pid_n * BLOCK + tl.arange(0, BLOCK)
    
    # Create masks for valid indices
    row_mask = row < m
    col_mask = col < n
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Initialize output element
    out = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    
    # Compute the reconstruction: U @ diag(s) @ Vh
    # For each element (i,j) of the output matrix:
    # out[i,j] = sum_k U[i,k] * s[k] * Vh[k,j]
    for k in range(k):
        # Load U[i,k] and Vh[k,j]
        u_val = tl.load(u_ptr + row * u_stride_0 + k * u_stride_1, mask=row_mask, other=0.0)
        s_val = tl.load(s_ptr + k * s_stride, mask=tl.arange(0, 1) < k, other=0.0)
        vh_val = tl.load(vh_ptr + k * vh_stride_0 + col * vh_stride_1, mask=col_mask, other=0.0)
        
        # Compute the contribution to the output
        u_s = u_val[:, None] * s_val[None, :]
        contribution = u_s * vh_val[None, :]
        out += contribution
    
    # Store the result
    tl.store(out_ptr + row * out_stride_0 + col * out_stride_1, out, mask=mask)

def fused_svd_reconstruct(A):
    # Perform SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Get dimensions
    m, n = A.shape
    k = S.shape[0]
    
    # Create output tensor
    out = torch.empty_like(A)
    
    # Launch kernel
    block = 16
    grid = (triton.cdiv(m, block), triton.cdiv(n, block))
    
    # Get strides
    u_stride_0, u_stride_1 = U.stride()
    s_stride = S.stride(0) if len(S.stride()) > 0 else 1
    vh_stride_0, vh_stride_1 = Vh.stride()
    out_stride_0, out_stride_1 = out.stride()
    
    _svd_reconstruct_kernel[grid](
        U, S, Vh, out,
        m, n, k,
        u_stride_0, u_stride_1,
        s_stride,
        vh_stride_0, vh_stride_1,
        out_stride_0, out_stride_1,
        BLOCK=block
    )
    
    return out
