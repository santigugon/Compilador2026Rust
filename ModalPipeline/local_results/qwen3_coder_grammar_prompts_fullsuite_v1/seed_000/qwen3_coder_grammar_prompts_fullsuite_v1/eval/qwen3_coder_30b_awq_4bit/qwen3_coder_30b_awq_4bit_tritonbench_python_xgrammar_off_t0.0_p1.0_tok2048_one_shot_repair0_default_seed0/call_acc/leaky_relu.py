import torch
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(
    input_ptr,
    output_ptr,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.where(input > 0, input, negative_slope * input)
    tl.store(output_ptr + offsets, output, mask=mask)

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    leaky_relu_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        negative_slope=negative_slope,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

##################################################################################################################################################



import torch

def test_leaky_relu():
    results = {}

    # Test case 1: Default parameters
    input_tensor_1 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_1"] = leaky_relu(input_tensor_1)

    # Test case 2: Custom negative_slope
    input_tensor_2 = torch.tensor([-2.0, 0.0, 2.0], device='cuda')
    results["test_case_2"] = leaky_relu(input_tensor_2, negative_slope=0.1)

    # Test case 3: Inplace operation
    input_tensor_3 = torch.tensor([-3.0, 0.0, 3.0], device='cuda')
    results["test_case_3"] = leaky_relu(input_tensor_3, inplace=True)

    # Test case 4: Larger tensor
    input_tensor_4 = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0], device='cuda')
    results["test_case_4"] = leaky_relu(input_tensor_4)

    return results

test_results = test_leaky_relu()
