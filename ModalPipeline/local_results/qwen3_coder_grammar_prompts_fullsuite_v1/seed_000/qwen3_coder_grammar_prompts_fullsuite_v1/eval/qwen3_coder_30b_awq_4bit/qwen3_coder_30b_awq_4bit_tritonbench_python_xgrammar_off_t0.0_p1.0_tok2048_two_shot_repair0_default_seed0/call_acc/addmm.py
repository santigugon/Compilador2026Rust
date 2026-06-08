import torch
import triton
import triton.language as tl

@triton.jit
def _addmm_kernel(mat1_ptr, mat2_ptr, input_ptr, out_ptr, 
                  m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
                  stride_mat1_m: tl.constexpr, stride_mat1_k: tl.constexpr,
                  stride_mat2_k: tl.constexpr, stride_mat2_n: tl.constexpr,
                  stride_input_m: tl.constexpr, stride_input_n: tl.constexpr,
                  stride_out_m: tl.constexpr, stride_out_n: tl.constexpr,
                  beta: tl.constexpr, alpha: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator for mat1 @ mat2
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, k, BLOCK_K):
        # Load mat1 and mat2 with appropriate strides
        mat1_ptrs = mat1_ptr + (offs_m[:, None] * stride_mat1_m + offs_k[None, :] * stride_mat1_k)
        mat2_ptrs = mat2_ptr + (offs_k[:, None] * stride_mat2_k + offs_n[None, :] * stride_mat2_n)
        
        # Load blocks
        mat1_block = tl.load(mat1_ptrs, mask=(offs_m[:, None] < m) & (offs_k[None, :] < k), other=0.0)
        mat2_block = tl.load(mat2_ptrs, mask=(offs_k[:, None] < k) & (offs_n[None, :] < n), other=0.0)
        
        # Matrix multiplication
        acc += tl.dot(mat1_block, mat2_block)
    
    # Scale and add input
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    
    # Load input
    input_ptrs = input_ptr + (offs_m[:, None] * stride_input_m + offs_n[None, :] * stride_input_n)
    input_block = tl.load(input_ptrs, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    
    # Compute output
    out_block = alpha * acc + beta * input_block
    
    # Store result
    tl.store(out_ptrs, out_block, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

def addmm(input, mat1, mat2, *, beta=1, alpha=1, out=None):
    # Ensure inputs are contiguous for easier handling
    mat1 = mat1.contiguous()
    mat2 = mat2.contiguous()
    input = input.contiguous()
    
    # Get dimensions
    m, k = mat1.shape
    k2, n = mat2.shape
    
    # Check dimensions match
    if k != k2:
        raise ValueError(f"mat1 and mat2 cannot be multiplied: {m}x{k} and {k2}x{n}")
    
    # Check if input is broadcastable with output shape (m, n)
    if input.shape != (m, n):
        # Try to broadcast
        try:
            torch.broadcast_tensors(input, torch.empty(m, n))
        except RuntimeError:
            raise ValueError(f"input shape {input.shape} is not broadcastable to {m}x{n}")
    
    # Create output tensor
    if out is None:
        out = torch.empty(m, n, dtype=mat1.dtype, device=mat1.device)
    else:
        if out.shape != (m, n):
            raise ValueError(f"out tensor must have shape {m}x{n}, got {out.shape}")
        if out.dtype != mat1.dtype or out.device != mat1.device:
            raise ValueError("out tensor must have the same dtype and device as mat1")
    
    # Launch kernel
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    grid_m = triton.cdiv(m, BLOCK_M)
    grid_n = triton.cdiv(n, BLOCK_N)
    grid = (grid_m, grid_n)
    
    # Get strides
    stride_mat1_m, stride_mat1_k = mat1.stride()
    stride_mat2_k, stride_mat2_n = mat2.stride()
    stride_input_m, stride_input_n = input.stride()
    stride_out_m, stride_out_n = out.stride()
    
    # Launch kernel
    _addmm_kernel[grid](
        mat1, mat2, input, out,
        m, n, k,
        stride_mat1_m, stride_mat1_k,
        stride_mat2_k, stride_mat2_n,
        stride_input_m, stride_input_n,
        stride_out_m, stride_out_n,
        beta, alpha,
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
