import torch
import triton
import triton.language as tl

@triton.jit
def _sqrt_exp_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sqrt_x = tl.sqrt(x)
    exp_sqrt_x = tl.exp(sqrt_x)
    tl.store(out_ptr + offsets, exp_sqrt_x, mask=mask)

def sqrt_exp(input, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input tensor"
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _sqrt_exp_kernel[grid](input, out, n, BLOCK=block)
    return out
