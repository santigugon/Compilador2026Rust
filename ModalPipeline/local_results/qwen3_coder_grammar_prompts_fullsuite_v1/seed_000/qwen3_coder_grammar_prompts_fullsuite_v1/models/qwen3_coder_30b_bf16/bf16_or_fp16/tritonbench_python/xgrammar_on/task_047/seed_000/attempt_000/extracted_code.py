import torch
import triton
import triton.language as tl

ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946

@triton.jit
def _selu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    exp_x = tl.exp(x)
    selu = SCALE * (tl.maximum(0, x) + tl.minimum(0, ALPHA * (exp_x - 1.0)))
    tl.store(out_ptr + offsets, selu, mask=mask)

def selu(input, inplace=False):
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _selu_kernel[grid](input, out, n, BLOCK=block)
    return out