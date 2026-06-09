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
    state = seed + offsets
    # Generate random numbers using LCG
    state = (state * 1103515245 + 12345) & 0x7fffffff
    # Convert to float in [0, 1)
    rand_val = state * (1.0 / 2147483648.0)
    tl.store(out_ptr + offsets, rand_val, mask=mask)

def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    # Handle the case where size is a single tuple/list
    if len(size) == 1 and not isinstance(size[0], int):
        size = size[0]
    
    # Determine the total number of elements
    total_elements = 1
    for s in size:
        total_elements *= s
    
    # Set default dtype if not provided
    if dtype is None:
        dtype = torch.get_default_dtype()
    
    # Set default device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output tensor
    if out is not None:
        if out.shape != torch.Size(size):
            raise ValueError("Output tensor shape does not match the specified size")
        if out.dtype != dtype:
            raise ValueError("Output tensor dtype does not match the specified dtype")
        if out.device != device:
            raise ValueError("Output tensor device does not match the specified device")
        out = out
    else:
        out = torch.empty(size, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad, pin_memory=pin_memory)
    
    # If size is 0, return empty tensor
    if total_elements == 0:
        return out
    
    # Get random seed
    seed = _get_random_seed(generator)
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(total_elements, block),)
    _rand_kernel[grid](out, total_elements, seed, BLOCK=block)
    
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
