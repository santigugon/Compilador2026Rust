import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    dim_size,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr
):
    # Compute the global thread index
    pid = tl.program_id(0)
    # Each block processes one slice along the specified dimension
    slice_size = num_elements // dim_size
    if slice_size == 0:
        slice_size = 1
    
    # Calculate the starting index for this block
    start_idx = pid * slice_size
    end_idx = min((pid + 1) * slice_size, num_elements)
    
    # Process each element in the slice
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        # Load data
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements
        if dtype == tl.float32:
            input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        else:
            input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(dtype)
        
        # Compute max and sum for softmax
        max_val = tl.max(input_vals, axis=0)
        exp_vals = tl.exp(input_vals - max_val)
        sum_exp = tl.sum(exp_vals, axis=0)
        
        # Compute softmax
        softmax_vals = exp_vals / sum_exp
        
        # Store result
        tl.store(output_ptr + offsets, softmax_vals, mask=mask)

def softmax(input, dim, dtype=None) -> torch.Tensor:
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Handle negative dimensions
    if dim < 0:
        dim = input.dim() + dim
    
    # Validate dimension
    if dim < 0 or dim >= input.dim():
        raise ValueError("Dimension out of range")
    
    # Cast input if dtype is specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Prepare output tensor
    output = torch.empty_like(input)
    
    # Get the size of the specified dimension
    dim_size = input.size(dim)
    
    # Get total number of elements
    num_elements = input.numel()
    
    # Create a view of the input tensor for processing
    input_view = input.contiguous().view(-1)
    output_view = output.contiguous().view(-1)
    
    # Launch kernel
    BLOCK_SIZE = 256
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Determine the appropriate data type for Triton
    triton_dtype = tl.float32 if input.dtype == torch.float32 else tl.float64
    
    # Launch kernel
    softmax_kernel[(num_blocks,)](
        input_view,
        output_view,
        dim_size,
        num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=triton_dtype
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
