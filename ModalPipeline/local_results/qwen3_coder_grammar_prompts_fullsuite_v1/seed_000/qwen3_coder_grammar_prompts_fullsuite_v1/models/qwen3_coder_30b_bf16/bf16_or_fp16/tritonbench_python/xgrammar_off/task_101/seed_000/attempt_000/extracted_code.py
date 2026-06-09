import torch
import triton
import triton.language as tl

@triton.jit
def digamma_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Handle special cases
    result = tl.where(x == 0.0, -tl.inf, 0.0)
    
    # For positive values, compute digamma using asymptotic expansion
    positive_mask = (x > 0.0) & (x != 0.0)
    x_pos = tl.load(x_ptr + offsets, mask=positive_mask)
    
    # Simple approximation for digamma function
    # For large x: digamma(x) ≈ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
    x_large = x_pos > 10.0
    x_small = ~x_large
    
    # Large x approximation
    x_large_val = tl.where(x_large, 
                          tl.log(x_pos) - 1.0/(2.0*x_pos) - 1.0/(12.0*x_pos*x_pos) + 
                          1.0/(120.0*x_pos*x_pos*x_pos*x_pos) - 
                          1.0/(252.0*x_pos*x_pos*x_pos*x_pos*x_pos*x_pos),
                          0.0)
    
    # Small x approximation using recurrence relation
    # digamma(x) = digamma(x+1) - 1/x for x < 1
    x_small_val = tl.where(x_small,
                          tl.where(x_pos > 1.0,
                                  tl.log(x_pos) - 1.0/(2.0*x_pos) - 1.0/(12.0*x_pos*x_pos) + 
                                  1.0/(120.0*x_pos*x_pos*x_pos*x_pos) - 
                                  1.0/(252.0*x_pos*x_pos*x_pos*x_pos*x_pos*x_pos),
                                  tl.log(x_pos + 1.0) - 1.0/(x_pos + 1.0) - 1.0/(2.0*x_pos) - 
                                  1.0/(12.0*x_pos*x_pos) + 1.0/(120.0*x_pos*x_pos*x_pos*x_pos) - 
                                  1.0/(252.0*x_pos*x_pos*x_pos*x_pos*x_pos*x_pos)),
                          0.0)
    
    result = tl.where(positive_mask, 
                     tl.where(x_pos > 10.0, x_large_val, x_small_val),
                     result)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def digamma(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    if input.numel() == 0:
        return out
    
    # Launch kernel
    grid = (triton.cdiv(input.numel(), 1024),)
    digamma_kernel[grid](
        input,
        out,
        input.numel(),
        BLOCK_SIZE=1024
    )
    
    return out
