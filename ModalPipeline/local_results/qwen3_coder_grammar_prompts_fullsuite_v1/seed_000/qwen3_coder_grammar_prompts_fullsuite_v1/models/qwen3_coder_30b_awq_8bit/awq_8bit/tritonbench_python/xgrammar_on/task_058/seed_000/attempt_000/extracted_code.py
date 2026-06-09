import torch
import triton
import triton.language as tl

def _clamp(x, min_val, max_val):
    return tl.minimum(tl.maximum(x, min_val), max_val)

@triton.jit
def _logit_kernel(x_ptr, out_ptr, n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Clamp input to [eps, 1-eps] if eps is not None
    if eps is not None:
        x = _clamp(x, eps, 1.0 - eps)
    else:
        # Clamp to [0, 1] and let NaN propagate for out-of-bounds
        x = _clamp(x, 0.0, 1.0)
        
    # Check for invalid values that would produce NaN
    # When x == 0 or x == 1, logit is undefined
    invalid = (x <= 0.0) | (x >= 1.0)
    
    # Compute logit: log(x / (1 - x))
    x_clamped = tl.where(invalid, 1e-8, x)  # Avoid division by zero
    logit_val = tl.log(x_clamped / (1.0 - x_clamped))
    
    # Set result to NaN for invalid cases
    result = tl.where(invalid, float('nan'), logit_val)
    
    tl.store(out_ptr + offsets, result, mask=mask)


def logit(input, eps=None, *, out=None):
    # Handle scalar input
    if not torch.is_tensor(input):
        input = torch.tensor(input)
        
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        
    # Get number of elements
    n = input.numel()
    
    # Handle empty tensor
    if n == 0:
        return out
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Convert eps to tensor scalar if provided
    eps_val = eps if eps is not None else None
    
    _logit_kernel[grid](input, out, n, eps_val, BLOCK=block)
    return out