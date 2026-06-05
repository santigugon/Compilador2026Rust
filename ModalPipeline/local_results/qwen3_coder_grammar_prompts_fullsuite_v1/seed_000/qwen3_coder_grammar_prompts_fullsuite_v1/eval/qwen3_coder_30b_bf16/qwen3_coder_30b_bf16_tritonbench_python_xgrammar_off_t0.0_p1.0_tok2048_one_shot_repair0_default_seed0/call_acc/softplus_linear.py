import torch
import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    n_cols, n_rows, beta, threshold,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    input_row = input_ptr + row * input_row_stride
    weight_row = weight_ptr + row * weight_row_stride
    output_row = output_ptr + row * output_row_stride
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = (i + tl.arange(0, BLOCK_SIZE)) < n_cols
        input_vals = tl.load(input_row + i + tl.arange(0, BLOCK_SIZE), mask=mask)
        weight_vals = tl.load(weight_row + i + tl.arange(0, BLOCK_SIZE), mask=mask)
        acc += input_vals * weight_vals
    
    if bias_ptr is not None:
        bias_row = bias_ptr + row * weight_row_stride
        bias_vals = tl.load(bias_row + col, mask=(col < n_cols))
        acc += bias_vals
    
    acc *= beta
    
    # Apply softplus with threshold
    softplus_val = tl.where(
        acc > threshold,
        acc,
        tl.log(1.0 + tl.exp(acc))
    )
    
    tl.store(output_row + col, softplus_val, mask=(col < n_cols))

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    assert input.dim() == 2, "Input must be a 2D tensor"
    assert weight.dim() == 2, "Weight must be a 2D tensor"
    assert input.size(1) == weight.size(1), "Input and weight dimensions must match"
    
    if bias is not None:
        assert bias.dim() == 1, "Bias must be a 1D tensor"
        assert bias.size(0) == weight.size(0), "Bias size must match weight output dimension"
    
    n_rows, n_cols = input.size()
    n_out = weight.size(0)
    
    output = torch.empty(n_rows, n_out, device=input.device, dtype=input.dtype)
    
    # Launch kernel
    grid = (n_rows, n_out)
    BLOCK_SIZE = 1024
    
    softplus_linear_kernel[grid](
        input, weight, bias, output,
        input.stride(0), weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        n_cols, n_rows, beta, threshold,
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
