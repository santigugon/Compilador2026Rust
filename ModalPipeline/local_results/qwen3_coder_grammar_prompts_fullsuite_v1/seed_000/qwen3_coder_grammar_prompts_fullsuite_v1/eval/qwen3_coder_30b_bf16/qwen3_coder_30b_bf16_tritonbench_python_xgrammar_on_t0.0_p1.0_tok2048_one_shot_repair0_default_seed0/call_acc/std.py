import torch
import triton
import triton.language as tl

def std(input, dim=None, *, correction=1, keepdim=False, out=None) -> torch.Tensor:
    if dim is None:
        dim = tuple(range(input.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    else:
        dim = tuple(dim)
    
    # Normalize dimensions
    dim = tuple(d if d >= 0 else d + input.ndim for d in dim)
    
    # Validate dimensions
    assert all(0 <= d < input.ndim for d in dim), "Invalid dimension specified"
    
    # Calculate output shape
    output_shape = list(input.shape)
    for d in sorted(dim, reverse=True):
        if keepdim:
            output_shape[d] = 1
        else:
            output_shape.pop(d)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == tuple(output_shape), "Output tensor shape mismatch"
        
    # Prepare for kernel launch
    input = input.contiguous()
    out = out.contiguous()
    
    # Launch kernel
    _std_kernel[(1,)](input, out, input.numel(), correction, keepdim, dim, input.ndim)
    
    return out

@triton.jit
def _std_kernel(input_ptr, output_ptr, n_elements, correction, keepdim, dim, ndim):
    # Get thread index
    pid = tl.program_id(0)
    
    # Calculate mean
    mean = 0.0
    for i in range(0, n_elements, 1):
        mean += tl.load(input_ptr + i)
    mean = mean / n_elements
    
    # Calculate variance
    variance = 0.0
    for i in range(0, n_elements, 1):
        val = tl.load(input_ptr + i)
        diff = val - mean
        variance += diff * diff
    
    # Apply correction
    if correction > 0:
        variance = variance / (n_elements - correction)
    else:
        variance = variance / n_elements
    
    # Calculate standard deviation
    std_val = tl.sqrt(variance)
    
    # Store result
    tl.store(output_ptr, std_val)
##################################################################################################################################################



import torch

def test_std():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_1"] = std(input_tensor)

    # Test case 2: Test with dim parameter
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_2"] = std(input_tensor, dim=0)

    # Test case 3: Test with keepdim=True
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_3"] = std(input_tensor, dim=1, keepdim=True)

    # Test case 4: Test with correction=0 (population standard deviation)
    input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    results["test_case_4"] = std(input_tensor, correction=0)

    return results

test_results = test_std()
