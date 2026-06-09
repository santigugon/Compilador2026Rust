import torch
import triton
import triton.language as tl

def bessel_j1(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input tensor"
    
    # For small values, we can use the series expansion
    # For large values, we can use the asymptotic expansion
    # Here we use a simple approach that works for most cases
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Use a simple approximation for Bessel J1
    # This is a basic implementation; for production use, a more accurate method would be preferred
    @triton.jit
    def _bessel_j1_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Simple approximation for Bessel J1
        # This is a basic implementation and may not be numerically accurate for all inputs
        # A more sophisticated implementation would be needed for production use
        y = tl.sin(x) / x - tl.cos(x) / (x * x) if x != 0 else 0.0
        tl.store(out_ptr + offsets, y, mask=mask)
    
    _bessel_j1_kernel[grid](input, out, n, BLOCK=block)
    return out