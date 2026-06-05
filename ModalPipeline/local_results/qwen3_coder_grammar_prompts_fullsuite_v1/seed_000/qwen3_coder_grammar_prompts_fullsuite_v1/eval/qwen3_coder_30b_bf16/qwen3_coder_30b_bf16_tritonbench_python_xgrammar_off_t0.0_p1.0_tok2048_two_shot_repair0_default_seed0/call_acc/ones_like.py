import torch
import triton
import triton.language as tl

@triton.jit
def _ones_like_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    # Load input to ensure we have the right shape and type for the output
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Create a tensor filled with 1.0
    ones = tl.full((BLOCK,), 1.0, dtype=x.dtype)
    # Store the result
    tl.store(out_ptr + offsets, ones, mask=mask)

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    # Handle dtype, layout, device, and memory_format
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    if layout is not None and layout != torch.strided:
        raise NotImplementedError("Only strided layout is supported")
    
    # Create output tensor with same shape as input but with specified dtype and device
    out = torch.empty_like(input, dtype=dtype, device=device, requires_grad=requires_grad)
    
    # If input is empty, return empty output
    if input.numel() == 0:
        return out
    
    # Launch kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _ones_like_kernel[grid](input, out, n, BLOCK=block)
    
    return out

##################################################################################################################################################



import torch

def test_ones_like():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_1"] = ones_like(input_tensor)

    # Test case 2: Test with a different dtype
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_2"] = ones_like(input_tensor, dtype=torch.float64)

    # Test case 3: Test with requires_grad=True
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_3"] = ones_like(input_tensor, requires_grad=True)

    # Test case 4: Test with a different device
    input_tensor = torch.randn(2, 3, device='cuda')
    results["test_case_4"] = ones_like(input_tensor, device='cuda')

    return results

test_results = test_ones_like()
