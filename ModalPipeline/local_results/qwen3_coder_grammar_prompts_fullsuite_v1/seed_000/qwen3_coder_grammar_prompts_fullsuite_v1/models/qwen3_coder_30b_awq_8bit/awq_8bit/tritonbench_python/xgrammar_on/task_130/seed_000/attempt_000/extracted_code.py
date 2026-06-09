import torch
import triton
import triton.language as tl
import math

def _get_random_seed(generator=None):
    if generator is not None:
        return generator.seed()
    else:
        return torch.seed()

@triton.jit
def _rand_kernel(out_ptr, n: tl.constexpr, seed: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    # Simple linear congruential generator
    # Using constants from Numerical Recipes
    a = 1664525
    c = 1013904223
    m = 2**32
    
    # Initialize state with seed and offset
    state = (seed + offsets) % m
    
    # Generate random numbers using LCG
    state = (a * state + c) % m
    
    # Convert to [0, 1) range
    rand_val = state / m
    
    tl.store(out_ptr + offsets, rand_val, mask=mask)

def rand(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    # Handle the case where size is a single tuple/list
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        shape = size[0]
    else:
        shape = size
    
    # Create the output tensor
    if out is not None:
        # Use provided output tensor
        if out.shape != shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match requested shape {shape}")
        if out.dtype != (dtype or torch.get_default_dtype()):
            raise ValueError(f"Output tensor dtype {out.dtype} does not match requested dtype {dtype}")
        if out.device != (device or torch.device('cpu')):
            raise ValueError(f"Output tensor device {out.device} does not match requested device {device}")
    else:
        # Create new tensor with specified properties
        out = torch.empty(shape, dtype=dtype or torch.get_default_dtype(), device=device or torch.device('cpu'), layout=layout, requires_grad=requires_grad, pin_memory=pin_memory)
    
    # Get the random seed
    seed = _get_random_seed(generator)
    
    # Calculate total number of elements
    n = out.numel()
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _rand_kernel[grid](out, n, seed, BLOCK=block)
    
    return out