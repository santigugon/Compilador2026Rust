import torch
import triton
import triton.language as tl

@triton.jit
def _selu_kernel(x_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, scale: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    exp_x = tl.exp(x)
    y = scale * (tl.maximum(0.0, x) + tl.minimum(0.0, alpha * (exp_x - 1.0)))
    tl.store(out_ptr + offsets, y, mask=mask)

def selu(input, inplace=False):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _selu_kernel[grid](input, out, n, alpha, scale, BLOCK=block)
    return out
