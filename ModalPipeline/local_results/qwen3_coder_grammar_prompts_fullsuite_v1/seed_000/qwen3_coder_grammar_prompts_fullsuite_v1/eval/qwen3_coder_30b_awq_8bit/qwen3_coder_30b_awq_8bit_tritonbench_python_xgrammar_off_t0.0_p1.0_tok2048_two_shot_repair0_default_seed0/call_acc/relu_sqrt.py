import torch
import triton
import triton.language as tl

@triton.jit
def _relu_sqrt_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Apply ReLU: set negative values to zero
    x = tl.maximum(x, 0.0)
    # Apply square root
    y = tl.sqrt(x)
    tl.store(out_ptr + offsets, y, mask=mask)

def relu_sqrt(input, inplace=False, out=None):
    if inplace:
        # For inplace operation, we need to ensure the input is contiguous
        # and modify it in place
        if not input.is_contiguous():
            input = input.contiguous()
        # Apply the operation in place
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _relu_sqrt_kernel[grid](input, input, n, BLOCK=block)
        return input
    else:
        # For non-inplace operation, use the provided out tensor or create a new one
        if out is not None:
            # Check if out tensor has the same shape as input
            if out.shape != input.shape:
                raise ValueError("Output tensor must have the same shape as input tensor")
            out = out.contiguous()
        else:
            out = torch.empty_like(input)
        
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _relu_sqrt_kernel[grid](input, out, n, BLOCK=block)
        return out

##################################################################################################################################################



import torch
from torch import Tensor

# def relu_sqrt(input: Tensor, inplace: bool=False, out: Tensor=None) -> Tensor:
#     if input.dtype != torch.float32 and input.dtype != torch.float64:
#         input = input.float()
#     if inplace:
#         input.relu_()
#         input.sqrt_()
#         return input
#     elif out is not None:
#         out.copy_(torch.sqrt(torch.relu(input)))
#         return out
#     else:
#         return torch.sqrt(torch.relu(input))

def test_relu_sqrt():
    results = {}
    
    # Test case 1: Default parameters
    a = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    results["test_case_1"] = relu_sqrt(a)
    
    # Test case 2: Inplace operation
    b = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    results["test_case_2"] = relu_sqrt(b, inplace=True)
    
    # Test case 3: Out parameter
    c = torch.tensor([-1.0, 0.0, 4.0, 9.0], device='cuda')
    out = torch.empty_like(c)
    results["test_case_3"] = relu_sqrt(c, out=out)
    
    # Test case 4: Non-float input
    d = torch.tensor([-1, 0, 4, 9], device='cuda')
    results["test_case_4"] = relu_sqrt(d)
    
    return results

test_results = test_relu_sqrt()
