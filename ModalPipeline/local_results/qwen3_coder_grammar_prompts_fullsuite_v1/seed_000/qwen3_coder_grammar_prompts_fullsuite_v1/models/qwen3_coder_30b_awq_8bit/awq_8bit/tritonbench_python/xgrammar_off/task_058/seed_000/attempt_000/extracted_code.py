import torch
import triton
import triton.language as tl

@triton.jit
def _logit_kernel(x_ptr, out_ptr, n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Clamp input to [eps, 1 - eps]
    if eps is not None:
        x = tl.clamp(x, eps, 1.0 - eps)
    
    # Compute logit: log(x / (1 - x))
    y = tl.log(x / (1.0 - x))
    
    tl.store(out_ptr + offsets, y, mask=mask)

def logit(input, eps=None, *, out=None):
    # Handle scalar input
    if not torch.is_tensor(input):
        if eps is not None:
            input = torch.clamp(input, eps, 1.0 - eps)
        return torch.log(input / (1.0 - input))
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input"
    
    # Handle special case where eps is None and input is outside [0, 1]
    if eps is None:
        # Check if any values are outside [0, 1] range
        if (input < 0).any() or (input > 1).any():
            # Fall back to PyTorch for NaN handling
            return torch.logit(input, eps=eps, out=out)
    
    # Launch kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Convert eps to a proper Triton constant
    eps_val = eps if eps is not None else 0.0
    
    _logit_kernel[grid](input, out, n, eps_val, BLOCK=block)
    return out
