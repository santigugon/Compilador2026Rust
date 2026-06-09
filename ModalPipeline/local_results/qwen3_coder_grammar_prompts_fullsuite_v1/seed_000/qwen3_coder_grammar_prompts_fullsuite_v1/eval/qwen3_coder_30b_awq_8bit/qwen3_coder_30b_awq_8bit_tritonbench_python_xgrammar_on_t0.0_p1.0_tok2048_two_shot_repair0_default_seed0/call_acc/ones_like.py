import torch
import triton
import triton.language as tl

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    # Handle the case where input is a scalar
    if input.dim() == 0:
        # For scalar inputs, we can directly return a scalar tensor
        if dtype is None:
            dtype = input.dtype
        out = torch.ones((), dtype=dtype, device=device, requires_grad=requires_grad)
        return out
    
    # Determine output properties
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    
    # Create output tensor with the same shape as input
    out = torch.empty_like(input, dtype=dtype, device=device, requires_grad=requires_grad)
    
    # Fill with ones using Triton kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    @triton.jit
    def _ones_like_kernel(out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        # Fill with ones
        tl.store(out_ptr + offsets, 1.0, mask=mask)
    
    _ones_like_kernel[grid](out, n, BLOCK=block)
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
