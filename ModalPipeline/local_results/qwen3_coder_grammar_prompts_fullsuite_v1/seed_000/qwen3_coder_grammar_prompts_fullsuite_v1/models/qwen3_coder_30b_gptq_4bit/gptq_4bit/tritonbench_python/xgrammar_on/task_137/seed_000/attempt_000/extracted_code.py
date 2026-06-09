import torch
import triton
import triton.language as tl

def leaky_relu(input, negative_slope=0.01, inplace=False):
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)
    
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    if inplace:
        _leaky_relu_kernel_inplace[grid](input, negative_slope, n, BLOCK=block)
    else:
        _leaky_relu_kernel[grid](input, out, negative_slope, n, BLOCK=block)
    
    return out

@triton.jit
def _leaky_relu_kernel(x_ptr, out_ptr, negative_slope: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Leaky ReLU: max(0, x) + negative_slope * min(0, x)
    y = tl.maximum(0, x) + negative_slope * tl.minimum(0, x)
    
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _leaky_relu_kernel_inplace(x_ptr, negative_slope: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Leaky ReLU: max(0, x) + negative_slope * min(0, x)
    y = tl.maximum(0, x) + negative_slope * tl.minimum(0, x)
    
    tl.store(x_ptr + offsets, y, mask=mask)