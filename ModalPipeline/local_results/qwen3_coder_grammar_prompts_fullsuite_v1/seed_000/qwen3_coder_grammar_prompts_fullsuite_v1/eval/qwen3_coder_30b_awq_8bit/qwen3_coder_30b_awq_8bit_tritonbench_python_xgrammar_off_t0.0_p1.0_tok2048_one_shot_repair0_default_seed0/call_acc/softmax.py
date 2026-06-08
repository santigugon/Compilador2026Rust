import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr, 
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr
):
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    
    input_row = input_ptr + row_id * n_cols
    output_row = output_ptr + row_id * n_cols
    
    # Load row data
    row = tl.load(input_row + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    
    # Apply softmax
    max_val = tl.max(row, axis=0)
    row_shifted = row - max_val
    exp_row = tl.exp(row_shifted)
    sum_exp = tl.sum(exp_row, axis=0)
    softmax_row = exp_row / sum_exp
    
    # Store result
    tl.store(output_row + tl.arange(0, BLOCK_SIZE), softmax_row, mask=tl.arange(0, BLOCK_SIZE) < n_cols)

def softmax(input, dim, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get input shape and compute total elements
    shape = input.shape
    n_rows = 1
    n_cols = 1
    
    # Compute dimensions for softmax
    for i in range(len(shape)):
        if i == dim:
            n_cols = shape[i]
        else:
            n_rows *= shape[i]
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (n_rows, 1, 1)
    
    softmax_kernel[grid](
        input,
        output,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=input.dtype
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
