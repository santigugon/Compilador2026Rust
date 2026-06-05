import torch
import triton
import triton.language as tl

@triton.jit
def addmm_kernel(
    input_ptr, mat1_ptr, mat2_ptr, out_ptr,
    input_stride_0, input_stride_1,
    mat1_stride_0, mat1_stride_1,
    mat2_stride_0, mat2_stride_1,
    out_stride_0, out_stride_1,
    n, m, p,
    beta, alpha,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + offs_m[:, None] * input_stride_0 + offs_n[None, :] * input_stride_1
    mat1_ptrs = mat1_ptr + offs_m[:, None] * mat1_stride_0 + offs_k[None, :] * mat1_stride_1
    mat2_ptrs = mat2_ptr + offs_k[:, None] * mat2_stride_0 + offs_n[None, :] * mat2_stride_1
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, m, BLOCK_SIZE_K):
        mat1 = tl.load(mat1_ptrs + k * mat1_stride_1, mask=(offs_k[None, :] + k < m))
        mat2 = tl.load(mat2_ptrs + k * mat2_stride_0, mask=(offs_k[:, None] + k < m))
        accumulator += tl.dot(mat1, mat2)
    
    out = tl.load(input_ptrs) * beta + accumulator * alpha
    
    out_ptrs = out_ptr + offs_m[:, None] * out_stride_0 + offs_n[None, :] * out_stride_1
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < n) & (offs_n[None, :] < p))

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    if mat1.dim() != 2 or mat2.dim() != 2:
        raise ValueError("mat1 and mat2 must be 2D tensors")
    
    n, m = mat1.shape
    m2, p = mat2.shape
    
    if m != m2:
        raise ValueError(f"mat1 and mat2 cannot be multiplied: {n}x{m} and {m2}x{p}")
    
    if input.shape != (n, p):
        raise ValueError(f"input shape {input.shape} is not broadcastable to {n}x{p}")
    
    if out is None:
        out = torch.empty(n, p, dtype=mat1.dtype, device=mat1.device)
    
    if input.dtype != mat1.dtype or input.dtype != mat2.dtype:
        raise ValueError("All tensors must have the same dtype")
    
    if mat1.device != mat2.device or mat1.device != input.device:
        raise ValueError("All tensors must be on the same device")
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    grid = (
        triton.cdiv(n, BLOCK_SIZE_M),
        triton.cdiv(p, BLOCK_SIZE_N)
    )
    
    addmm_kernel[grid](
        input, mat1, mat2, out,
        input.stride(0), input.stride(1),
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1),
        n, m, p,
        beta, alpha,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out
