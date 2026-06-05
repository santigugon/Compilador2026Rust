import torch
import triton
import triton.language as tl

@triton.jit
def erf_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: int = 1024):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute erf using approximation
    # Based on Abramowitz and Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413788
    a4 = -1.453159026
    a5 = 1.061405429
    p = 0.3275911
    
    sign = tl.where(x < 0, -1.0, 1.0)
    x = tl.abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * tl.exp(-x * x)
    
    result = sign * y
    tl.store(output_ptr + offsets, result, mask=mask)

def erf(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    
    erf_kernel[grid](input, out, n_elements)
    
    return out
