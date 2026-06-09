import torch
import triton
import triton.language as tl

def _clamp_input(x, eps):
    if eps is not None:
        return tl.minimum(tl.maximum(x, eps), 1.0 - eps)
    return x

@triton.jit
def _logit_kernel(x_ptr, out_ptr, n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = _clamp_input(x, eps)
    # Avoid division by zero
    x = tl.where(x <= 0.0 or x >= 1.0, 0.0, x)
    y = tl.log(x / (1.0 - x))
    tl.store(out_ptr + offsets, y, mask=mask)


def logit(input, eps=None, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Handle scalar input
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    
    # Convert eps to a tensor for kernel
    eps_val = eps if eps is not None else 0.0
    
    _logit_kernel[grid](input, out, n, eps_val, BLOCK=block)
    return out