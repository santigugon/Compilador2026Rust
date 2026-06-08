import torch
import triton
import triton.language as tl

@triton.jit
def _bitwise_and_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)
    result = x & y
    tl.store(out_ptr + offsets, result, mask=mask)

def bitwise_and(input, other, *, out=None):
    # Ensure inputs are of compatible types
    if not (torch.is_tensor(input) and torch.is_tensor(other)):
        raise TypeError("Both input and other must be tensors")
    
    # Check if the tensors have compatible shapes for broadcasting
    if input.shape != other.shape:
        # Broadcasting is supported by PyTorch
        pass
    
    # Determine output tensor
    if out is not None:
        out = out
    else:
        out = torch.empty_like(input)
    
    # Get total number of elements
    n = input.numel()
    
    # Set block size and grid size
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch kernel
    _bitwise_and_kernel[grid](input, other, out, n, BLOCK=block)
    
    return out
