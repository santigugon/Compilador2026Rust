import torch
import triton
import triton.language as tl

def _matmul_kernel(x_ptr, w_ptr, b_ptr, out_ptr, M, N, K, stride_xm, stride_xk, stride_wn, stride_wk, stride_bm, stride_bn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the starting indices for this block
    m_start = pid * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load tiles
        x_tile = tl.load(x_ptr + (m_start + tl.arange(0, BLOCK_M)[:, None]) * stride_xm + (k + tl.arange(0, BLOCK_K)[None, :]) * stride_xk, mask=(m_start + tl.arange(0, BLOCK_M)[:, None]) < M)
        w_tile = tl.load(w_ptr + (n_start + tl.arange(0, BLOCK_N)[:, None]) * stride_wn + (k + tl.arange(0, BLOCK_K)[None, :]) * stride_wk, mask=(n_start + tl.arange(0, BLOCK_N)[:, None]) < N)
        
        # Matrix multiplication
        acc += tl.dot(x_tile, w_tile.T)
    
    # Apply bias if provided
    if b_ptr is not None:
        bias = tl.load(b_ptr + (n_start + tl.arange(0, BLOCK_N)) * stride_bn, mask=(n_start + tl.arange(0, BLOCK_N)) < N)
        acc += bias[None, :]
    
    # Apply tanh
    acc = 2.0 / (1.0 + tl.exp(-2.0 * acc)) - 1.0
    
    # Store result
    tl.store(out_ptr + (m_start + tl.arange(0, BLOCK_M)[:, None]) * stride_xm + (n_start + tl.arange(0, BLOCK_N)) * stride_xk, acc, mask=(m_start + tl.arange(0, BLOCK_M)[:, None]) < M)


def tanh_linear(input, weight, bias=None):
    # Ensure input is 2D for matrix multiplication
    original_shape = input.shape
    input_2d = input.view(-1, input.size(-1))
    
    # Get dimensions
    M, K = input_2d.shape
    N, _ = weight.shape
    
    # Create output tensor
    out = torch.empty(M, N, dtype=input.dtype, device=input.device)
    
    # Define block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32
    
    # Grid dimensions
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (grid_m, grid_n)
    
    # Launch kernel
    _matmul_kernel[grid](
        input_2d, weight, bias, out,
        M, N, K,
        input_2d.stride(0), input_2d.stride(1),
        weight.stride(0), weight.stride(1),
        bias.stride(0) if bias is not None else 0, 1,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    # Reshape output to original shape
    return out.view(original_shape[:-1] + (-1,))
##################################################################################################################################################



import torch
from tanh_linear import tanh_linear

def test_tanh_linear():
    results = {}

    # Test case 1: input, weight, and bias on GPU
    input1 = torch.randn(5, 3, device='cuda')
    weight1 = torch.randn(4, 3, device='cuda')
    bias1 = torch.randn(4, device='cuda')
    result1 = tanh_linear(input1, weight1, bias1)
    results["test_case_1"] = result1

    # Test case 2: input and weight on GPU, bias is None
    input2 = torch.randn(5, 3, device='cuda')
    weight2 = torch.randn(4, 3, device='cuda')
    result2 = tanh_linear(input2, weight2)
    results["test_case_2"] = result2

    # Test case 3: input and weight on GPU, bias on GPU
    input3 = torch.randn(2, 3, device='cuda')
    weight3 = torch.randn(2, 3, device='cuda')
    bias3 = torch.randn(2, device='cuda')
    result3 = tanh_linear(input3, weight3, bias3)
    results["test_case_3"] = result3

    # Test case 4: input, weight, and bias on GPU with different dimensions
    input4 = torch.randn(3, 2, device='cuda')
    weight4 = torch.randn(2, 2, device='cuda')
    bias4 = torch.randn(2, device='cuda')
    result4 = tanh_linear(input4, weight4, bias4)
    results["test_case_4"] = result4

    return results

test_results = test_tanh_linear()
