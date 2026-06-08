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
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, m, BLOCK_SIZE_K):
        mat1 = tl.load(mat1_ptr + offs_m[:, None] * mat1_stride_0 + offs_k[None, :] * mat1_stride_1)
        mat2 = tl.load(mat2_ptr + offs_k[:, None] * mat2_stride_0 + offs_n[None, :] * mat2_stride_1)
        acc += tl.dot(mat1, mat2)
    
    acc = acc * alpha
    
    if beta != 0:
        input = tl.load(input_ptr + offs_m[:, None] * input_stride_0 + offs_n[None, :] * input_stride_1)
        acc += input * beta
    
    tl.store(out_ptr + offs_m[:, None] * out_stride_0 + offs_n[None, :] * out_stride_1, acc)

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    if out is None:
        out = torch.empty(mat1.size(0), mat2.size(1), dtype=mat1.dtype, device=mat1.device)
    
    n, m = mat1.shape
    m2, p = mat2.shape
    
    if n != out.size(0) or p != out.size(1):
        raise ValueError("Output tensor dimensions don't match expected (n x p)")
    
    if m != m2:
        raise ValueError("Matrix dimensions don't match for multiplication")
    
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

##################################################################################################################################################



import torch

def test_addmm():
    results = {}

    # Test case 1: Default beta and alpha
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_1"] = addmm(input1, mat1_1, mat2_1)

    # Test case 2: Custom beta and alpha
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_2"] = addmm(input2, mat1_2, mat2_2, beta=0.5, alpha=2.0)

    # Test case 3: Zero beta
    input3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    mat1_3 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    mat2_3 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    results["test_case_3"] = addmm(input3, mat1_3, mat2_3, beta=0.0)

    return results

test_results = test_addmm()
