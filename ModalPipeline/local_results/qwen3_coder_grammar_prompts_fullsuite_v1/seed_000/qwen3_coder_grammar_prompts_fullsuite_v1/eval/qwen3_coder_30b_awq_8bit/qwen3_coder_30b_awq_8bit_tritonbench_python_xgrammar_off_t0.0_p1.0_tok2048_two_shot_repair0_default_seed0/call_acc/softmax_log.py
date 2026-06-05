import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_log_kernel(x_ptr, out_ptr, dim_size: tl.constexpr, n_elements: tl.constexpr, stride_x: tl.constexpr, stride_out: tl.constexpr, BLOCK: tl.constexpr):
    # Get the program ID for the dimension we're processing
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this block
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets * stride_x, mask=mask, other=0.0)
    
    # Apply log
    x = tl.log(x)
    
    # Apply softmax along the specified dimension
    # First, find the maximum value for numerical stability
    max_val = tl.max(x, axis=0)
    x = x - max_val
    
    # Compute exp and sum
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    
    # Normalize
    out = exp_x / sum_exp_x
    
    # Store result
    tl.store(out_ptr + offsets * stride_out, out, mask=mask)

def softmax_log(input, dim=-1, dtype=None):
    # Handle dtype casting if specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Ensure input is contiguous for easier handling
    input = input.contiguous()
    
    # Get output tensor with same shape as input
    out = torch.empty_like(input)
    
    # Get the size of the specified dimension
    dim_size = input.size(dim)
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Calculate block size
    BLOCK = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK),)
    
    # Get strides for the input and output tensors
    input_strides = input.stride()
    output_strides = out.stride()
    
    # Get stride for the specified dimension
    stride_x = input_strides[dim] if dim >= 0 else input_strides[input.dim() + dim]
    stride_out = output_strides[dim] if dim >= 0 else output_strides[input.dim() + dim]
    
    # Launch kernel
    _softmax_log_kernel[grid](
        input, out, dim_size, n_elements, stride_x, stride_out, BLOCK=BLOCK
    )
    
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def softmax_log(input, dim=-1, dtype=None):
#     if dtype is not None:
#         input = input.to(dtype)
#     log_input = input.log()
#     return F.softmax(log_input, dim=dim)

def test_softmax_log():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = softmax_log(input_tensor)

    # Test case 2: Specifying a different dimension
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = softmax_log(input_tensor, dim=0)

    # Test case 3: Specifying a different dtype
    input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_3"] = softmax_log(input_tensor, dtype=torch.float64)

    # Test case 4: Larger tensor
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_4"] = softmax_log(input_tensor)

    return results

test_results = test_softmax_log()
