import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-input))
    tl.store(output_ptr + offsets, sigmoid_val, mask=mask)

@triton.jit
def argmax_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    max_val = tl.max(input, axis=0)
    max_idx = tl.arg_max(input, axis=0)
    tl.store(output_ptr + offsets, max_idx, mask=mask)

def sigmoid_argmax(input, dim=None, keepdim=False):
    if dim is None:
        # Flatten the tensor for global argmax
        input_flat = input.flatten()
        n_elements = input_flat.numel()
        output = torch.empty(n_elements, dtype=torch.float32, device=input.device)
        
        # Launch sigmoid kernel
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        sigmoid_kernel[grid](input_flat, output, n_elements, BLOCK_SIZE)
        
        # Compute argmax
        max_val, max_idx = torch.max(output, dim=0)
        return max_idx
    else:
        # For specific dimension, we need to handle differently
        # This is a simplified version that assumes dim=0 for demonstration
        input_copy = input.clone()
        # Apply sigmoid
        input_copy = 1.0 / (1.0 + torch.exp(-input_copy))
        # Compute argmax along specified dimension
        max_val, max_idx = torch.max(input_copy, dim=dim, keepdim=keepdim)
        return max_idx

##################################################################################################################################################



import torch

def test_sigmoid_argmax():
    results = {}

    # Test case 1: 1D tensor, no dim specified
    input1 = torch.tensor([0.1, 2.0, -1.0, 3.0], device='cuda')
    results["test_case_1"] = sigmoid_argmax(input1)

    # Test case 2: 2D tensor, dim=0
    input2 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_2"] = sigmoid_argmax(input2, dim=0)

    # Test case 3: 2D tensor, dim=1
    input3 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_3"] = sigmoid_argmax(input3, dim=1)

    # Test case 4: 2D tensor, dim=1, keepdim=True
    input4 = torch.tensor([[0.1, 2.0, -1.0], [3.0, -0.5, 1.5]], device='cuda')
    results["test_case_4"] = sigmoid_argmax(input4, dim=1, keepdim=True)

    return results

test_results = test_sigmoid_argmax()
