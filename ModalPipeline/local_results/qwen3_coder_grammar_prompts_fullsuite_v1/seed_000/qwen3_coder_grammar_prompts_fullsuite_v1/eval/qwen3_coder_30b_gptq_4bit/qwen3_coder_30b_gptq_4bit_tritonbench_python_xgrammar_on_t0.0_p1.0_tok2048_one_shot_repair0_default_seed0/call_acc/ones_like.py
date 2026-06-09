import torch
import triton
import triton.language as tl

def ones_like_kernel(input_ptr, output_ptr, size, dtype, BLOCK_SIZE=1024):
    """Triton kernel to fill output tensor with ones"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    
    # Fill with ones
    ones = tl.full((BLOCK_SIZE,), 1, dtype=dtype)
    tl.store(output_ptr + offsets, ones, mask=mask)

@triton.jit
def ones_like_kernel_jit(input_ptr, output_ptr, size, dtype, BLOCK_SIZE=1024):
    """Triton kernel to fill output tensor with ones"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    
    # Fill with ones
    ones = tl.full((BLOCK_SIZE,), 1, dtype=dtype)
    tl.store(output_ptr + offsets, ones, mask=mask)

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    """
    Returns a tensor filled with the scalar value 1, with the same size as the input tensor.
    """
    # Get input tensor properties
    input_size = input.numel()
    input_dtype = input.dtype
    input_device = input.device
    
    # Determine output properties
    output_dtype = dtype if dtype is not None else input_dtype
    output_device = device if device is not None else input_device
    
    # Create output tensor
    output = torch.ones(input.shape, dtype=output_dtype, device=output_device, requires_grad=requires_grad)
    
    # Handle memory format if needed
    if memory_format != torch.preserve_format:
        output = output.contiguous(memory_format=memory_format)
    
    return output
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
