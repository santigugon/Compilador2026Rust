import torch
import triton
import triton.language as tl

@triton.jit
def selu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    alpha: tl.constexpr,
    scale: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    exp_input = tl.exp(input)
    selu_val = scale * tl.where(input >= 0, input, alpha * (exp_input - 1))
    tl.store(output_ptr + offsets, selu_val, mask=mask)

def selu(input, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    alpha = 1.6732632423543774
    scale = 1.0507009873554804
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    selu_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        alpha=alpha,
        scale=scale
    )
    
    return output

##################################################################################################################################################



def test_selu():
    # Initialize a dictionary to store test results
    results = {}

    # Test case 1: Positive values
    input_tensor_1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = selu(input_tensor_1)

    # Test case 2: Negative values
    input_tensor_2 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_2"] = selu(input_tensor_2)

    # Test case 3: Mixed values
    input_tensor_3 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_3"] = selu(input_tensor_3)

    # Test case 4: Zero values
    input_tensor_4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    results["test_case_4"] = selu(input_tensor_4)

    return results

test_results = test_selu()
