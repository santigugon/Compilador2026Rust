import torch
import triton
import triton.language as tl

@triton.jit
def _addmm_kernel(input_ptr, mat1_ptr, mat2_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, p: tl.constexpr, beta: tl.constexpr, alpha: tl.constexpr, input_stride_0: tl.constexpr, input_stride_1: tl.constexpr, mat1_stride_0: tl.constexpr, mat1_stride_1: tl.constexpr, mat2_stride_0: tl.constexpr, mat2_stride_1: tl.constexpr, out_stride_0: tl.constexpr, out_stride_1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the starting indices for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create masks for valid indices
    mask_m = offs_m < n
    mask_n = offs_n < p
    mask_k = offs_k < m
    
    # Load input matrix
    input = tl.load(input_ptr + offs_m[:, None] * input_stride_0 + offs_n[None, :] * input_stride_1, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # Initialize output with input
    out = input * beta
    
    # Compute mat1 @ mat2
    for k in range(0, m, BLOCK_K):
        # Load blocks of mat1 and mat2
        mat1_block = tl.load(mat1_ptr + offs_m[:, None] * mat1_stride_0 + (k + offs_k[None, :]) * mat1_stride_1, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        mat2_block = tl.load(mat2_ptr + (k + offs_k[:, None]) * mat2_stride_0 + offs_n[None, :] * mat2_stride_1, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        # Compute partial dot product
        partial = tl.dot(mat1_block, mat2_block)
        
        # Accumulate into output
        out += alpha * partial
    
    # Store result
    tl.store(out_ptr + offs_m[:, None] * out_stride_0 + offs_n[None, :] * out_stride_1, out, mask=mask_m[:, None] & mask_n[None, :])

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    # Handle scalar alpha and beta
    if not isinstance(alpha, (int, float)):
        alpha = alpha.item()
    if not isinstance(beta, (int, float)):
        beta = beta.item()
    
    # Get dimensions
    n, m = mat1.shape
    m2, p = mat2.shape
    
    # Check if dimensions are compatible
    if m != m2:
        raise ValueError("mat1 and mat2 must have compatible dimensions for matrix multiplication")
    
    # Check if input is broadcastable with output
    if input.shape != (n, p):
        # Try to broadcast input
        try:
            input = input.expand(n, p)
        except RuntimeError:
            raise ValueError("input tensor is not broadcastable with the expected output shape")
    
    # Create output tensor
    if out is not None:
        out = torch.empty_like(out)
    else:
        out = torch.empty(n, p, dtype=mat1.dtype, device=mat1.device)
    
    # Determine block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    # Calculate grid size
    grid_m = triton.cdiv(n, BLOCK_M)
    grid_n = triton.cdiv(p, BLOCK_N)
    grid = (grid_m, grid_n)
    
    # Launch kernel
    _addmm_kernel[grid](
        input, mat1, mat2, out,
        n, m, p,
        beta, alpha,
        input.stride(0), input.stride(1),
        mat1.stride(0), mat1.stride(1),
        mat2.stride(0), mat2.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
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
