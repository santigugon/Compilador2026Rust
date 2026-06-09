import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _rad2deg_sqrt_kernel(x_ptr, out1_ptr, out2_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Convert radians to degrees
    y1 = x * 180.0 / 3.141592653589793
    # Calculate square root
    y2 = tl.sqrt(x)
    tl.store(out1_ptr + offsets, y1, mask=mask)
    tl.store(out2_ptr + offsets, y2, mask=mask)

def rad2deg_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out1 = torch.empty_like(input)
    out2 = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _rad2deg_sqrt_kernel[grid](input, out1, out2, n, BLOCK=block)
    return (out1, out2)