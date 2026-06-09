import torch
import triton
import triton.language as tl
import math

@triton.jit
def _rand_kernel(out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    # Generate random numbers using a simple linear congruential generator
    # Using a fixed seed for reproducibility
    seed = 12345
    # Simple hash-like function to generate pseudo-random numbers
    # This is a basic implementation - in practice, a more sophisticated PRNG would be used
    x = (offsets * 1103515245 + 12345) & 0x7fffffff
    # Convert to float in [0, 1) range
    y = x * (1.0 / (2**31))
    tl.store(out_ptr + offsets, y, mask=mask)

def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    # Handle the case where size is passed as a single tuple/list
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    
    # Convert size to a tuple if it's not already
    if not isinstance(size, tuple):
        size = tuple(size)
    
    # Determine the total number of elements
    total_elements = 1
    for dim in size:
        total_elements *= dim
    
    # Set default device and dtype
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Create the output tensor
    if out is not None:
        # Validate that out has the correct shape and dtype
        if out.shape != size or out.dtype != dtype:
            raise ValueError("out tensor must have the same shape and dtype as requested")
        out = out.to(device)
    else:
        out = torch.empty(size, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad, pin_memory=pin_memory)
    
    # If the tensor is empty, return it immediately
    if total_elements == 0:
        return out
    
    # Launch the kernel
    block = 256
    grid = (triton.cdiv(total_elements, block),)
    _rand_kernel[grid](out, total_elements, BLOCK=block)
    
    return out
