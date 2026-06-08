import torch
import triton
import triton.language as tl

@triton.jit
def _selu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # SELU constants
    alpha = 1.6732632423543772848170429452695
    scale = 1.0507009873554804934193349852946
    
    # SELU computation: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    exp_x = tl.exp(x)
    term1 = scale * tl.maximum(0, x)
    term2 = scale * tl.minimum(0, alpha * (exp_x - 1.0))
    result = term1 + term2
    
    tl.store(out_ptr + offsets, result, mask=mask)

def selu(input, inplace=False):
    if inplace:
        raise NotImplementedError("Inplace SELU is not implemented in this version")
    
    out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _selu_kernel[grid](input, out, n, BLOCK=block)
    return out
