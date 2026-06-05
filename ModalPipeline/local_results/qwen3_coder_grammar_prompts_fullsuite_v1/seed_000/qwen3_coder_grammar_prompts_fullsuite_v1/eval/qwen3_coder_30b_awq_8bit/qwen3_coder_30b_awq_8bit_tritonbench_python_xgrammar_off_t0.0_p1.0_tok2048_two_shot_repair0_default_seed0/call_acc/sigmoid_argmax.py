import torch
import triton
import triton.language as tl

@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _argmax_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # For argmax, we need to find the maximum value and its index
    # This is a simplified version that works for the flattened case
    # In a real implementation, we'd need to handle the dimension properly
    max_val = tl.max(x)
    max_idx = tl.argmin(tl.where(x == max_val, 0, 1))
    tl.store(out_ptr, max_idx, mask=mask)

def sigmoid_argmax(input, dim=None, keepdim=False):
    # Apply sigmoid to input
    sigmoid_input = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _sigmoid_kernel[grid](input, sigmoid_input, n, BLOCK=block)
    
    # Compute argmax
    if dim is None:
        # Flatten and find argmax over all elements
        flat_input = sigmoid_input.flatten()
        result = torch.argmax(flat_input)
        if keepdim:
            result = result.view(1)
    else:
        # Find argmax along specified dimension
        result = torch.argmax(sigmoid_input, dim=dim, keepdim=keepdim)
    
    return result

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
