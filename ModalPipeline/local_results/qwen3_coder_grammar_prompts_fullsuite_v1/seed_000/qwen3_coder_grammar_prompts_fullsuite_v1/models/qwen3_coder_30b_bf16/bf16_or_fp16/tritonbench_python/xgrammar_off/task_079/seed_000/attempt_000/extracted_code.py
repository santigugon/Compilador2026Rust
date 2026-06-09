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
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(p, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    mat1_ptrs = mat1_ptr + offs_m[:, None] * mat1_stride_0 + offs_k[None, :] * mat1_stride_1
    mat2_ptrs = mat2_ptr + offs_k[:, None] * mat2_stride_0 + offs_n[None, :] * mat2_stride_1
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(m, BLOCK_SIZE_K)):
        a = tl.load(mat1_ptrs)
        b = tl.load(mat2_ptrs)
        accumulator += tl.dot(a, b)
        mat1_ptrs += BLOCK_SIZE_K * mat1_stride_1
        mat2_ptrs += BLOCK_SIZE_K * mat2_stride_0
    
    out_ptrs = out_ptr + offs_m[:, None] * out_stride_0 + offs_n[None, :] * out_stride_1
    input_ptrs = input_ptr + offs_m[:, None] * input_stride_0 + offs_n[None, :] * input_stride_1
    
    out = tl.load(out_ptrs)
    input_val = tl.load(input_ptrs)
    
    result = alpha * accumulator + beta * input_val
    
    tl.store(out_ptrs, result)

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    if out is None:
        out = torch.empty(mat1.size(0), mat2.size(1), dtype=mat1.dtype, device=mat1.device)
    
    n, m = mat1.shape
    m2, p = mat2.shape
    
    if n != out.size(0) or p != out.size(1):
        raise ValueError("Output tensor dimensions don't match expected (n, p)")
    
    if m != m2:
        raise ValueError("Matrix dimensions don't match for multiplication")
    
    if input.size(0) != n or input.size(1) != p:
        raise ValueError("Input tensor dimensions don't match expected (n, p)")
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    num_warps = 4
    num_stages = 4
    
    grid = (triton.cdiv(n, BLOCK_SIZE_M) * triton.cdiv(p, BLOCK_SIZE_N),)
    
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
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=num_warps,
        num_stages=num_stages
    )
    
    return out
