import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_idx * n_cols + col_offsets
    mask = col_offsets < n_cols
    
    # Load input values
    input_vals = tl.load(input_ptrs, mask=mask)
    
    # Numerical stability: subtract max
    max_val = tl.max(input_vals, axis=0)
    input_vals = input_vals - max_val
    
    # Compute exp
    exp_vals = tl.exp(input_vals)
    
    # Compute sum
    sum_val = tl.sum(exp_vals, axis=0)
    
    # Compute softmax
    output_vals = exp_vals / sum_val
    
    # Store result
    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, output_vals, mask=mask)


def softmax(input, dim, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get output shape
    output_shape = input.shape
    
    # Flatten input to 2D for kernel processing
    if dim < 0:
        dim = len(input.shape) + dim
    
    # Compute the size of the dimension we're softmaxing over
    softmax_dim_size = input.shape[dim]
    
    # Compute the number of slices
    num_slices = 1
    for i in range(len(input.shape)):
        if i != dim:
            num_slices *= input.shape[i]
    
    # Reshape to 2D
    input_2d = input.view(num_slices, softmax_dim_size)
    
    # Create output tensor
    output = torch.empty_like(input_2d)
    
    # Launch kernel
    grid = (num_slices,)
    BLOCK_SIZE = triton.next_power_of_2(softmax_dim_size)
    softmax_kernel[grid](input_2d, output, softmax_dim_size, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape back to original shape
    output = output.view(output_shape)
    
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
