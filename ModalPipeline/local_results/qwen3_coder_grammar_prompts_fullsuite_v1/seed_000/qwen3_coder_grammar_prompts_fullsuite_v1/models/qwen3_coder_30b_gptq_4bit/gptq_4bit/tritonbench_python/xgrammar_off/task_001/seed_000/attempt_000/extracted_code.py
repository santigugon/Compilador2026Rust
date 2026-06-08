import torch
import triton
import triton.language as tl

@triton.jit
def _div_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, rounding_mode: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform division
    result = x / y
    
    # Apply rounding if specified
    if rounding_mode == "floor":
        result = tl.floor(result)
    elif rounding_mode == "trunc":
        result = tl.trunc(result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure both tensors have the same device
    if other.device != input.device:
        other = other.to(input.device)
    
    # Handle broadcasting
    out_shape = torch.broadcast_tensors(input, other)[0].shape
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    
    # Determine block size and grid
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Prepare pointers
    x_ptr = input.data_ptr()
    y_ptr = other.data_ptr()
    out_ptr = out.data_ptr()
    
    # Launch kernel
    _div_kernel[grid](x_ptr, y_ptr, out_ptr, n, rounding_mode, BLOCK=block)
    
    return out
