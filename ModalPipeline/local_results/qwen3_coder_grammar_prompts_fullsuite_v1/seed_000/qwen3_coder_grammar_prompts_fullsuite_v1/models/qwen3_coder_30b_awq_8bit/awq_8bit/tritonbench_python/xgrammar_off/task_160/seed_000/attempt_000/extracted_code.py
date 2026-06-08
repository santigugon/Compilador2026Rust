import torch
import triton
import triton.language as tl

@triton.jit
def _ones_like_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    # Load input to ensure we have the right shape and stride info
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Store ones with the same shape and stride
    tl.store(out_ptr + offsets, 1.0, mask=mask)

def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format):
    # Handle dtype, layout, device parameters
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    if layout is None:
        layout = input.layout
    
    # Create output tensor with same shape as input but with specified dtype and device
    out = torch.empty_like(input, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)
    
    # If input is empty, return empty output
    if input.numel() == 0:
        return out
    
    # Launch kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _ones_like_kernel[grid](input, out, n, BLOCK=block)
    
    return out
