import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(x_ptr, out_ptr, dim_size: tl.constexpr, stride_x, stride_out, BLOCK: tl.constexpr):
    # Get the row index
    row_idx = tl.program_id(0)
    
    # Load data for this row
    offsets = row_idx * stride_x + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets, mask=offsets < dim_size, other=-float('inf'))
    
    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    
    # Compute exp
    x_exp = tl.exp(x)
    
    # Compute sum
    x_sum = tl.sum(x_exp, axis=0)
    
    # Compute softmax
    softmax = x_exp / x_sum
    
    # Store result
    out_offsets = row_idx * stride_out + tl.arange(0, BLOCK)
    tl.store(out_ptr + out_offsets, softmax, mask=out_offsets < dim_size)

def softmax(input, dim, dtype=None):
    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
    
    # Get input shape and ensure it's at least 1D
    input_shape = input.shape
    if len(input_shape) == 0:
        input = input.unsqueeze(0)
    
    # Get the size of the specified dimension
    dim_size = input_shape[dim]
    
    # Get the total number of elements in the tensor
    total_elements = input.numel()
    
    # Create output tensor
    out = torch.empty_like(input)
    
    # Handle the case where we need to compute softmax along a dimension
    # For simplicity, we'll compute softmax along the last dimension
    # If dim is not the last dimension, we need to transpose appropriately
    
    # For this implementation, we'll compute softmax along the specified dimension
    # by treating each slice along that dimension as a separate vector
    
    # Compute the number of slices along the specified dimension
    # We'll compute softmax for each slice along the specified dimension
    if dim < 0:
        dim = len(input_shape) + dim
    
    # Compute the number of elements in each slice
    slice_size = 1
    for i in range(dim + 1, len(input_shape)):
        slice_size *= input_shape[i]
    
    # Compute the number of slices
    num_slices = total_elements // slice_size
    
    # Create a temporary tensor for the computation
    if dim == len(input_shape) - 1:
        # If we're computing softmax along the last dimension
        block = 256
        grid = (num_slices,)
        _softmax_kernel[grid](
            input,
            out,
            dim_size,
            input.stride(dim),
            out.stride(dim),
            BLOCK=block
        )
    else:
        # For other dimensions, we need to handle transposition
        # This is a simplified approach - for full generality, we'd need more complex logic
        # But for the basic case, we'll compute along the last dimension
        # and transpose if needed
        input_transposed = input.transpose(dim, -1)
        out_transposed = out.transpose(dim, -1)
        
        block = 256
        grid = (num_slices,)
        _softmax_kernel[grid](
            input_transposed,
            out_transposed,
            dim_size,
            input_transposed.stride(-1),
            out_transposed.stride(-1),
            BLOCK=block
        )
        
        # Transpose back
        out = out_transposed.transpose(dim, -1)
    
    # If we originally had a scalar input, squeeze it back
    if len(input_shape) == 0:
        out = out.squeeze(0)
    
    return out

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
