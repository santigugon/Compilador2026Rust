import torch
import triton
import triton.language as tl

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    # Perform linear transformation
    if bias is not None:
        output = torch.nn.functional.linear(input, weight, bias)
    else:
        output = torch.nn.functional.linear(input, weight)
    
    # Apply softplus activation
    out = torch.empty_like(output)
    
    # Get output tensor properties
    n = output.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Launch kernel
    _softplus_linear_kernel[grid](output, out, n, beta, threshold, BLOCK=block)
    
    return out

@triton.jit
def _softplus_linear_kernel(x_ptr, out_ptr, n: tl.constexpr, beta: tl.constexpr, threshold: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute softplus: log(1 + exp(beta * x)) if x <= threshold, else x
    # For numerical stability, we use the identity: log(1 + exp(x)) = x + log(1 + exp(-x)) when x > 0
    # But we'll use a more direct approach
    # If x > threshold, use linear function (x)
    # If x <= threshold, use softplus function
    
    # Create a mask for values <= threshold
    mask_le = x <= threshold
    
    # Compute softplus for values <= threshold
    softplus_val = tl.log(1.0 + tl.exp(beta * x))
    
    # For values > threshold, use linear function
    linear_val = x
    
    # Select between softplus and linear based on threshold
    result = tl.where(mask_le, softplus_val, linear_val)
    
    tl.store(out_ptr + offsets, result, mask=mask)