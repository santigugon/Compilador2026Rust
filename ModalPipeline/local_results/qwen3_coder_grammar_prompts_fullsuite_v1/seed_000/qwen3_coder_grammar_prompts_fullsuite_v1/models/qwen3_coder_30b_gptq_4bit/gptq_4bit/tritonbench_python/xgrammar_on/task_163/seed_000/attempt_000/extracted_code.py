import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def _cos_signbit_kernel(x_ptr, cos_out_ptr, signbit_out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute cosine
    cos_x = tl.cos(x)
    # Compute sign bit (1 if negative, 0 if positive or zero)
    signbit = tl.where(cos_x < 0, 1.0, 0.0)
    
    tl.store(cos_out_ptr + offsets, cos_x, mask=mask)
    tl.store(signbit_out_ptr + offsets, signbit, mask=mask)

def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(input)
    signbit_out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _cos_signbit_kernel[grid](input, out, signbit_out, n, BLOCK=block)
    return out, signbit_out