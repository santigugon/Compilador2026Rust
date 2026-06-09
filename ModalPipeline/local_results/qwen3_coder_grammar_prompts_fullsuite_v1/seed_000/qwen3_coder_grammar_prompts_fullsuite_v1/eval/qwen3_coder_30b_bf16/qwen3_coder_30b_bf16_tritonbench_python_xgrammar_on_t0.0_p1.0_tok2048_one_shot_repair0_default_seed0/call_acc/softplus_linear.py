import torch
import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    input_ptr,  # pointer to input tensor
    weight_ptr,  # pointer to weight tensor
    bias_ptr,  # pointer to bias tensor (optional)
    output_ptr,  # pointer to output tensor
    input_row_stride,
    weight_row_stride,
    weight_col_stride,
    output_row_stride,
    n_cols,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the row index
    row = tl.program_id(0)
    
    # Compute the input row pointer
    input_row_ptr = input_ptr + row * input_row_stride
    
    # Compute the output row pointer
    output_row_ptr = output_ptr + row * output_row_stride
    
    # Initialize accumulator for linear transformation
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Perform linear transformation
    for col in range(0, n_cols, BLOCK_SIZE):
        # Load input values
        input_vals = tl.load(input_row_ptr + col * tl.arange(0, BLOCK_SIZE))
        
        # Load weight values
        weight_vals = tl.load(weight_ptr + col * weight_col_stride + tl.arange(0, BLOCK_SIZE))
        
        # Accumulate
        accumulator += input_vals * weight_vals
    
    # Add bias if present
    if bias_ptr is not None:
        bias_row_ptr = bias_ptr
        for col in range(0, n_cols, BLOCK_SIZE):
            bias_vals = tl.load(bias_row_ptr + col * tl.arange(0, BLOCK_SIZE))
            accumulator += bias_vals
    
    # Apply Softplus activation
    # Softplus(x) = (1/beta) * log(1 + exp(beta * x))
    # For numerical stability, when x > threshold, we use x instead of softplus
    softplus_vals = tl.where(
        accumulator > threshold,
        accumulator,
        (1.0 / beta) * tl.log(1.0 + tl.exp(beta * accumulator))
    )
    
    # Store the result
    tl.store(output_row_ptr + tl.arange(0, BLOCK_SIZE), softplus_vals)


def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Ensure weight is contiguous
    weight = weight.contiguous()
    
    # Get dimensions
    n_rows, n_cols = input.shape
    _, n_out = weight.shape
    
    # Create output tensor
    output = torch.empty((n_rows, n_out), device=input.device, dtype=input.dtype)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = (n_rows,)
    
    # Determine if bias is provided
    if bias is not None:
        bias_ptr = bias
    else:
        bias_ptr = None
    
    # Launch kernel
    softplus_linear_kernel[grid](
        input, weight, bias_ptr, output,
        input.stride(0), weight.stride(0), weight.stride(1),
        output.stride(0),
        n_cols,
        beta,
        threshold,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
##################################################################################################################################################



import torch

def test_softplus_linear():
    results = {}

    # Test case 1: Basic test with default beta and threshold
    input1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight1 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias1 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_1"] = softplus_linear(input1, weight1, bias1)

    # Test case 2: Test with non-default beta
    input2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight2 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias2 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_2"] = softplus_linear(input2, weight2, bias2, beta=2)

    # Test case 3: Test with non-default threshold
    input3 = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device='cuda')
    weight3 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    bias3 = torch.tensor([0.0, 0.0], device='cuda')
    results["test_case_3"] = softplus_linear(input3, weight3, bias3, threshold=15)

    # Test case 4: Test with no bias
    input4 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    weight4 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device='cuda')
    results["test_case_4"] = softplus_linear(input4, weight4)

    return results

test_results = test_softplus_linear()
