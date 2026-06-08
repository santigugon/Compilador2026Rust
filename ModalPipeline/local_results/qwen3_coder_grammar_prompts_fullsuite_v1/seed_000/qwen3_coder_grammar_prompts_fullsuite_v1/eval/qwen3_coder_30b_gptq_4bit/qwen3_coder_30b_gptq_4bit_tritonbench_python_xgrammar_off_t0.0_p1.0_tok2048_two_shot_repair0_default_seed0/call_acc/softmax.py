import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    dim_size: tl.constexpr, 
    BLOCK: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this block
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute softmax
    # First, subtract the maximum for numerical stability
    max_val = tl.max(input_data, axis=0)
    input_data = input_data - max_val
    
    # Compute exponentials
    exp_data = tl.exp(input_data)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_data, axis=0)
    
    # Compute softmax
    softmax_data = exp_data / sum_exp
    
    # Store result
    tl.store(output_ptr + offsets, softmax_data, mask=mask)

def softmax(input, dim, dtype=None):
    # Handle dtype casting if specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Get the shape of the input tensor
    shape = input.shape
    dim_size = shape[dim]
    total_elements = input.numel()
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Determine block size and grid size
    BLOCK = 256
    grid = (triton.cdiv(total_elements, BLOCK),)
    
    # Launch kernel
    _softmax_kernel[grid](
        input, 
        output, 
        total_elements, 
        dim_size, 
        BLOCK=BLOCK
    )
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def softmax(input: torch.Tensor, dim: int, dtype: torch.dtype=None) -> torch.Tensor:
#     return F.softmax(input, dim=dim, dtype=dtype)

def test_softmax():
    results = {}
    
    # Test case 1: Basic test with default dtype
    input1 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_1"] = softmax(input1, dim=1)
    
    # Test case 2: Test with different dimension
    input2 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_2"] = softmax(input2, dim=0)
    
    # Test case 3: Test with specified dtype
    input3 = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    results["test_case_3"] = softmax(input3, dim=1, dtype=torch.float64)
    
    # Test case 4: Test with larger tensor
    input4 = torch.randn(100, 100, device='cuda')
    results["test_case_4"] = softmax(input4, dim=1)
    
    return results

test_results = test_softmax()
