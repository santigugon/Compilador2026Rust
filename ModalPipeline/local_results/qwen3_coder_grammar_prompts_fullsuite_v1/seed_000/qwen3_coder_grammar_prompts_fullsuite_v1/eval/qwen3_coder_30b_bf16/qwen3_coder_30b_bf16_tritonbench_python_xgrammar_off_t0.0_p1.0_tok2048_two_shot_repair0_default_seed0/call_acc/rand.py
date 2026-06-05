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
    # Using constants from the standard LCG formula
    state = offsets + 1  # Initialize with offset to avoid zero
    # Simple hash-like transformation to get pseudo-random numbers
    state = (state * 1103515245 + 12345) & 0x7fffffff
    # Convert to float in [0, 1) range
    rand_val = state * (1.0 / 2147483648.0)
    tl.store(out_ptr + offsets, rand_val, mask=mask)

def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    # Handle the case where size is a single integer or a sequence
    if len(size) == 1 and not isinstance(size[0], int):
        # size is a sequence like a list or tuple
        shape = size[0]
    else:
        shape = size
    
    # Convert shape to a tuple if it's not already
    if not isinstance(shape, tuple):
        shape = tuple(shape)
    
    # Calculate total number of elements
    n = 1
    for dim in shape:
        n *= dim
    
    # Set default dtype if not provided
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Create output tensor
    if out is not None:
        # Validate that out has the correct shape and dtype
        if out.shape != shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {shape}")
        if out.dtype != dtype:
            raise ValueError(f"Output tensor dtype {out.dtype} does not match expected dtype {dtype}")
        out = out
    else:
        out = torch.empty(shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    
    # If the tensor is empty, return it immediately
    if n == 0:
        return out
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    _rand_kernel[grid](out, n, BLOCK=block)
    
    return out

##################################################################################################################################################



import torch

def test_rand():
    results = {}

    # Test case 1: Basic usage with size only
    results["test_case_1"] = rand(2, 3, device='cuda')

    # Test case 2: Specifying dtype
    results["test_case_2"] = rand(2, 3, dtype=torch.float64, device='cuda')

    # Test case 3: Using a generator
    gen = torch.Generator(device='cuda')
    gen.manual_seed(42)
    results["test_case_3"] = rand(2, 3, generator=gen, device='cuda')

    # Test case 4: Requires gradient
    results["test_case_4"] = rand(2, 3, requires_grad=True, device='cuda')

    return results

test_results = test_rand()
