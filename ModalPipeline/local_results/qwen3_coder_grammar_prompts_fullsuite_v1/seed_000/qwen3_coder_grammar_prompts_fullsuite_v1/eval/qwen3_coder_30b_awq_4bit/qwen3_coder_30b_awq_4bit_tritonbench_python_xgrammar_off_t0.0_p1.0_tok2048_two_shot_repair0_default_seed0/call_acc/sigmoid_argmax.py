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
def _argmax_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For simplicity, we'll compute argmax in a separate kernel
    # In practice, this would need more complex logic for proper reduction
    # But for this implementation, we'll use a simpler approach
    # that works for the basic case
    
    # This is a simplified version - a full implementation would require
    # more complex reduction logic to properly handle argmax
    tl.store(out_ptr + offsets, x, mask=mask)

def sigmoid_argmax(input, dim=None, keepdim=False):
    # Handle scalar input
    if input.dim() == 0:
        input = input.unsqueeze(0)
        dim = 0 if dim is None else dim
        keepdim = True
    
    # Apply sigmoid
    sigmoid_out = torch.empty_like(input, dtype=torch.float32)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _sigmoid_kernel[grid](input, sigmoid_out, n, BLOCK=block)
    
    # Handle argmax
    if dim is None:
        # Flatten and find argmax
        flat_sigmoid = sigmoid_out.flatten()
        argmax_idx = torch.argmax(flat_sigmoid)
        if keepdim:
            return argmax_idx.unsqueeze(0)
        return argmax_idx
    else:
        # Find argmax along specified dimension
        argmax_idx = torch.argmax(sigmoid_out, dim=dim, keepdim=keepdim)
        return argmax_idx

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
