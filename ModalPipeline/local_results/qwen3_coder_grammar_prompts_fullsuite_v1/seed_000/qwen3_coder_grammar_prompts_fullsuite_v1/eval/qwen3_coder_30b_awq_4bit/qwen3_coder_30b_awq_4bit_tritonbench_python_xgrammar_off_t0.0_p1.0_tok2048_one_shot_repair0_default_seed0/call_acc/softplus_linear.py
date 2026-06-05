import torch
import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row_stride, weight_row_stride, weight_col_stride,
    output_row_stride, output_col_stride,
    n_rows, n_cols, threshold, beta,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    if row >= n_rows:
        return
    
    input_row = input_ptr + row * input_row_stride
    output_row = output_ptr + row * output_row_stride
    
    for col in range(0, n_cols, BLOCK_SIZE):
        col_offset = col + tl.arange(0, BLOCK_SIZE)
        mask = col_offset < n_cols
        
        input_vals = tl.load(input_row + col_offset * 1, mask=mask)
        weight_vals = tl.load(weight_ptr + col_offset * weight_col_stride, mask=mask)
        output_vals = tl.sum(input_vals * weight_vals, axis=0)
        
        if bias_ptr is not None:
            bias_vals = tl.load(bias_ptr + col_offset * 1, mask=mask)
            output_vals += bias_vals
        
        # Apply softplus
        output_vals = tl.where(
            output_vals > threshold,
            output_vals,
            tl.log(1.0 + tl.exp(beta * output_vals)) / beta
        )
        
        tl.store(output_row + col_offset * output_col_stride, output_vals, mask=mask)

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    input = input.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    n_rows, n_cols = input.shape
    output = torch.empty(n_rows, weight.shape[0], dtype=input.dtype, device=input.device)
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_rows, 1),)
    
    softplus_linear_kernel[grid](
        input, weight, bias, output,
        input.stride(0), weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
        n_rows, n_cols, threshold, beta,
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
