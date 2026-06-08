import torch
import triton
import triton.language as tl
import math

@triton.jit
def _logspace_kernel(
    start_ptr, end_ptr, out_ptr, 
    steps: tl.constexpr, 
    base: tl.constexpr, 
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < steps
    
    # Load start and end values (they are scalars)
    start_val = tl.load(start_ptr)
    end_val = tl.load(end_ptr)
    
    # Compute the logarithmic spacing
    # We want: base^start, base^(start + step), ..., base^end
    # where step = (end - start) / (steps - 1) for steps > 1
    # For steps = 1, we just return base^start
    
    if steps == 1:
        # Special case: only one element
        result = tl.pow(base, start_val)
        tl.store(out_ptr + offsets, result, mask=mask)
    else:
        # General case: evenly spaced logarithmic values
        step = (end_val - start_val) / (steps - 1)
        indices = start_val + offsets * step
        result = tl.pow(base, indices)
        tl.store(out_ptr + offsets, result, mask=mask)

def logspace(start, end, steps, base=10.0, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
    # Handle scalar inputs
    if not torch.is_tensor(start):
        start = torch.tensor(start, dtype=torch.float32)
    if not torch.is_tensor(end):
        end = torch.tensor(end, dtype=torch.float32)
    
    # Validate steps
    if steps <= 0:
        raise ValueError("steps must be a positive integer")
    
    # Determine output tensor properties
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Create output tensor
    if out is not None:
        out = torch.empty_like(out, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)
    else:
        out = torch.empty(steps, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)
    
    # Handle special case of steps = 1
    if steps == 1:
        if base == 10.0:
            out[0] = 10 ** start.item()
        else:
            out[0] = base ** start.item()
        return out
    
    # For normal case, launch kernel
    block = 256
    grid = (triton.cdiv(steps, block),)
    
    # Convert to appropriate types for kernel
    start_val = start.item() if torch.is_tensor(start) else start
    end_val = end.item() if torch.is_tensor(end) else end
    
    # Create temporary tensors for kernel
    start_tensor = torch.tensor(start_val, dtype=torch.float32)
    end_tensor = torch.tensor(end_val, dtype=torch.float32)
    
    _logspace_kernel[grid](start_tensor, end_tensor, out, steps, base, BLOCK=block)
    
    return out
