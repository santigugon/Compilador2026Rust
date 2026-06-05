import torch
import triton
import triton.language as tl
import math

@triton.jit
def _rand_kernel(out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Simple linear congruential generator (LCG) parameters
    # These are standard values that provide good distribution properties
    a = 1664525
    c = 1013904223
    m = 2**32
    
    # Initialize seed with thread index
    seed = offsets % m
    
    # Generate pseudo-random numbers using LCG
    # We use a simple approach to avoid complex operations
    rand_vals = (a * seed + c) % m
    # Normalize to [0, 1) range
    normalized = rand_vals.to(tl.float32) / m.to(tl.float32)
    
    tl.store(out_ptr + offsets, normalized, mask=mask)

def rand(*size, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False):
    # Handle variable size arguments
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        shape = tuple(size[0])
    else:
        shape = size
    
    # Create output tensor with specified shape
    if out is not None:
        # If out is provided, use it
        if out.shape != shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match requested shape {shape}")
        if dtype is not None and out.dtype != dtype:
            raise ValueError(f"Output tensor dtype {out.dtype} does not match requested dtype {dtype}")
        if device is not None and out.device != device:
            raise ValueError(f"Output tensor device {out.device} does not match requested device {device}")
        result = out
    else:
        # Create new tensor with specified properties
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        result = torch.empty(shape, dtype=dtype, device=device, layout=layout, requires_grad=requires_grad)
    
    # Handle pin_memory
    if pin_memory and device.type == 'cpu':
        result = result.pin_memory()
    
    # Fill with random numbers
    n = result.numel()
    if n > 0:
        block = 256
        grid = (triton.cdiv(n, block),)
        _rand_kernel[grid](result, n, BLOCK=block)
    
    return result

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
