import torch
import triton
import triton.language as tl

def _matmul_kernel(mat1_ptr, mat2_ptr, input_ptr, out_ptr, m, n, k, beta, alpha, input_stride_0, input_stride_1, mat1_stride_0, mat1_stride_1, mat2_stride_0, mat2_stride_1, out_stride_0, out_stride_1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the block offsets
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Load input
    input_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    input = tl.load(input_ptr + offs_m[:, None] * input_stride_0 + offs_n[None, :] * input_stride_1, mask=input_mask, other=0.0)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute mat1 @ mat2
    for i in range(0, k, BLOCK_K):
        # Load blocks
        mat1_mask = (offs_m[:, None] < m) & (offs_k[None, :] < k - i)
        mat2_mask = (offs_k[:, None] < k - i) & (offs_n[None, :] < n)
        
        mat1 = tl.load(mat1_ptr + offs_m[:, None] * mat1_stride_0 + (offs_k[None, :] + i) * mat1_stride_1, mask=mat1_mask, other=0.0)
        mat2 = tl.load(mat2_ptr + (offs_k[:, None] + i) * mat2_stride_0 + offs_n[None, :] * mat2_stride_1, mask=mat2_mask, other=0.0)
        
        # Compute partial dot product
        acc += tl.dot(mat1, mat2)
    
    # Apply scaling
    acc = alpha * acc + beta * input
    
    # Store result
    out_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(out_ptr + offs_m[:, None] * out_stride_0 + offs_n[None, :] * out_stride_1, acc, mask=out_mask)


def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    # Ensure inputs are contiguous
    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    input = input.contiguous()
    
    # Get dimensions
    m, k = mat1.shape
    k2, n = mat2.shape
    
    # Check dimensions
    if k != k2:
        raise ValueError(f"mat1 and mat2 cannot be multiplied: {m}x{k} and {k2}x{n}")
    
    # Broadcast input to match output shape
    if input.shape != (m, n):
        input = input.expand(m, n)
    
    # Create output tensor
    if out is None:
        out = torch.empty(m, n, dtype=mat1.dtype, device=mat1.device)
    else:
        if out.shape != (m, n):
            raise ValueError(f"out tensor must have shape ({m}, {n}), got {out.shape}")
        if out.dtype != mat1.dtype or out.device != mat1.device:
            raise ValueError("out tensor must have the same dtype and device as mat1")
    
    # Define block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Launch kernel
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    
    _matmul_kernel[grid](
        mat1, mat2, input, out,
        m, n, k,
        beta, alpha,
        input.stride(0), input.stride(1),
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return out