import triton
import triton.language as tl

def rsqrt(input, *, out=None):
    if out is None:
        out = input.new_empty(input.shape)
    
    # Get the number of elements in the input tensor
    num_elements = input.numel()
    
    # Define block size
    block_size = 1024
    
    # Launch kernel
    grid = (num_elements + block_size - 1) // block_size
    _rsqrt_kernel[grid, block_size](input, out, num_elements)
    
    return out

@triton.jit
def _rsqrt_kernel(input_ptr, output_ptr, num_elements):
    # Compute the index of the current thread
    idx = tl.program_id(0) * tl.constexpr(1024) + tl.arange(0, 1024)
    
    # Create mask for valid elements
    mask = idx < num_elements
    
    # Load input data
    input = tl.load(input_ptr + idx, mask=mask)
    
    # Compute reciprocal square root
    output = tl.rsqrt(input)
    
    # Store result
    tl.store(output_ptr + idx, output, mask=mask)
##################################################################################################################################################



import torch

def test_rsqrt():
    results = {}

    # Test case 1: Positive elements
    input1 = torch.tensor([4.0, 16.0, 25.0], device='cuda')
    results["test_case_1"] = rsqrt(input1)

    # Test case 2: Contains zero
    input2 = torch.tensor([0.0, 1.0, 4.0], device='cuda')
    results["test_case_2"] = rsqrt(input2)

    # Test case 3: Contains negative elements
    input3 = torch.tensor([-1.0, 4.0, 9.0], device='cuda')
    results["test_case_3"] = rsqrt(input3)

    # Test case 4: All elements are zero
    input4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = rsqrt(input4)

    return results

test_results = test_rsqrt()
