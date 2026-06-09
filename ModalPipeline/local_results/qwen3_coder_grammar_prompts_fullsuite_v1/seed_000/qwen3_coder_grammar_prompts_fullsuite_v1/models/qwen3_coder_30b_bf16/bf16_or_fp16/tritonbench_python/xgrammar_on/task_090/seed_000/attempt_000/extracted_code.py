import torch
import triton
import triton.language as tl
import math

def fused_hardshrink_dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False, lambd: float = 0.5) -> torch.Tensor:
    if not training:
        # If not in training mode, just apply hard shrinkage
        out = torch.empty_like(input)
        
        @triton.jit
        def _hardshrink_kernel(x_ptr, out_ptr, n: tl.constexpr, lambd: tl.constexpr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            # Hard shrinkage: if |x| <= lambd, return 0; otherwise return x
            condition = tl.abs(x) <= lambd
            y = tl.where(condition, 0.0, x)
            tl.store(out_ptr + offsets, y, mask=mask)
        
        n = input.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _hardshrink_kernel[grid](input, out, n, lambd, BLOCK=block)
        return out
    
    # In training mode, apply dropout first
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)
    
    # Apply dropout
    if training:
        # Generate random mask
        mask = torch.rand_like(input) > p
        out = input * mask
    else:
        out = input.clone()
    
    # Apply hard shrinkage
    @triton.jit
    def _hardshrink_kernel(x_ptr, out_ptr, n: tl.constexpr, lambd: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        # Hard shrinkage: if |x| <= lambd, return 0; otherwise return x
        condition = tl.abs(x) <= lambd
        y = tl.where(condition, 0.0, x)
        tl.store(out_ptr + offsets, y, mask=mask)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _hardshrink_kernel[grid](out, out, n, lambd, BLOCK=block)
    return out