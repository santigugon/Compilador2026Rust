import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(x_ptr, out_ptr, stride_x, stride_out, dim_size: tl.constexpr, BLOCK: tl.constexpr):
    # Get the program ID for the dimension we're processing
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this block
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Create mask for valid elements
    mask = offsets < dim_size
    
    # Load input data
    x = tl.load(x_ptr + offsets * stride_x, mask=mask, other=-float('inf'))
    
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
    tl.store(out_ptr + offsets * stride_out, softmax, mask=mask)

def softmax(input, dim, dtype=None):
    # Handle dtype casting if specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Create output tensor with same shape and dtype
    out = torch.empty_like(input)
    
    # Get the size of the specified dimension
    dim_size = input.shape[dim]
    
    # Get the stride for the specified dimension
    stride = input.stride(dim)
    
    # For small dimensions, use a simple approach
    if dim_size <= 1024:
        block = 256
        grid = (triton.cdiv(dim_size, block),)
        
        # Create a temporary tensor for the computation
        temp_input = input.contiguous()
        temp_out = out.contiguous()
        
        _softmax_kernel[grid](
            temp_input,
            temp_out,
            stride,
            temp_out.stride(dim),
            dim_size,
            BLOCK=block
        )
    else:
        # For larger dimensions, we need to handle it differently
        # This is a simplified approach that works for most cases
        # For a more robust implementation, we'd need to handle
        # multi-dimensional cases more carefully
        
        # Reshape to handle the softmax along the specified dimension
        # We'll compute softmax along the last dimension for simplicity
        # and then reshape back
        input_flat = input.view(-1, dim_size)
        out_flat = out.view(-1, dim_size)
        
        for i in range(input_flat.shape[0]):
            # Compute softmax for each slice
            slice_input = input_flat[i]
            slice_out = out_flat[i]
            
            # Get max for numerical stability
            max_val = slice_input.max()
            exp_vals = (slice_input - max_val).exp()
            sum_vals = exp_vals.sum()
            slice_out.copy_(exp_vals / sum_vals)
    
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
