import triton
import triton.language as tl

def log_kernel(input, output, n_elements, BLOCK_SIZE=1024):
    # Compute the natural logarithm using Triton
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_blocks, 1, 1)
    log_kernel_impl[grid](input, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

@triton.jit
def log_kernel_impl(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the block index
    block_idx = tl.program_id(0)
    # Compute the start and end indices for this block
    start_idx = block_idx * BLOCK_SIZE
    end_idx = tl.minimum(start_idx + BLOCK_SIZE, n_elements)
    # Create a mask for valid elements
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load input data
    input = tl.load(input_ptr + offsets, mask=mask)
    # Compute natural logarithm
    output = tl.log(input)
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def log(input, *, out=None):
    # Get the number of elements in the input tensor
    n_elements = input.numel()
    # Create output tensor if not provided
    if out is None:
        out = input.new_empty(input.shape)
    # Launch the kernel
    log_kernel(input, out, n_elements)
    return out
##################################################################################################################################################



import torch

def test_log():
    results = {}

    # Test case 1: Basic test with positive values
    input1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    results["test_case_1"] = log(input1)

    # Test case 2: Test with a tensor containing a zero
    input2 = torch.tensor([0.0, 1.0, 2.0], device='cuda')
    results["test_case_2"] = log(input2)

    # Test case 3: Test with a tensor containing negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_3"] = log(input3)

    # Test case 4: Test with a tensor containing a mix of positive, negative, and zero
    input4 = torch.tensor([-1.0, 0.0, 1.0], device='cuda')
    results["test_case_4"] = log(input4)

    return results

test_results = test_log()
