import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _rad2deg_sqrt_kernel(x_ptr, deg_ptr, sqrt_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Convert radians to degrees: degrees = radians * (180 / pi)
    deg = x * (180.0 / 3.141592653589793)
    
    # Calculate square root
    sqrt = tl.sqrt(x)
    
    tl.store(deg_ptr + offsets, deg, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out_deg = torch.empty_like(input)
    out_sqrt = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _rad2deg_sqrt_kernel[grid](input, out_deg, out_sqrt, n, BLOCK=block)
    return (out_deg, out_sqrt)
