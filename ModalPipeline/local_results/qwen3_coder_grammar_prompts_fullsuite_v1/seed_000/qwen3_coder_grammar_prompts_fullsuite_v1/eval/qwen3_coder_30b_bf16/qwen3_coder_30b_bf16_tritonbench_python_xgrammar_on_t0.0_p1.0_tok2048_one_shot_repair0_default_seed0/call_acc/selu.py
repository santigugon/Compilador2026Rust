import torch
import triton
import triton.language as tl

@triton.jit
def selu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the block index
    block_idx = tl.program_id(0)
    # Compute the start index for this block
    start_idx = block_idx * BLOCK_SIZE
    # Create a mask for valid elements
    mask = start_idx + tl.arange(0, BLOCK_SIZE) < n_elements
    # Load data
    x = tl.load(x_ptr + start_idx + tl.arange(0, BLOCK_SIZE), mask=mask)
    # SELU computation: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    alpha = 1.6733
    scale = 1.0507
    exp_x = tl.exp(x)
    term1 = tl.maximum(0.0, x)
    term2 = tl.minimum(0.0, alpha * (exp_x - 1.0))
    result = scale * (term1 + term2)
    # Store result
    tl.store(output_ptr + start_idx + tl.arange(0, BLOCK_SIZE), result, mask=mask)

def selu(input, inplace=False):
    # Ensure input is contiguous
    input = input.contiguous()
    # Create output tensor
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    # Get the number of elements
    n_elements = input.numel()
    # Launch the kernel
    grid = (triton.cdiv(n_elements, 1024),)
    selu_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
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
