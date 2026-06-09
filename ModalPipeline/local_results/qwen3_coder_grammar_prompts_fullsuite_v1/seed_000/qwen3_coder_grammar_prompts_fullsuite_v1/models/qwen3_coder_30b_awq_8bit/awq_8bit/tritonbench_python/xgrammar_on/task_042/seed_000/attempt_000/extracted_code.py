import torch
import triton
import triton.language as tl

@triton.jit
def _zeta_kernel(x_ptr, q_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0)
    
    # Initialize sum
    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    
    # Compute the series sum: sum_{n=0}^{inf} 1 / (q + n)^x
    # We'll use a reasonable number of terms for approximation
    # For better accuracy, we could increase num_terms
    num_terms = 100
    
    # Handle special case where x <= 0
    # For x <= 0, the series doesn't converge in the usual sense
    # We'll use a simple approximation or return 0
    x_is_positive = x > 0.0
    
    # For x > 0, compute the series
    for n in range(num_terms):
        n_val = tl.cast(n, tl.float32)
        denominator = q + n_val
        # Avoid division by zero
        denominator = tl.where(denominator == 0.0, 1e-10, denominator)
        term = tl.pow(denominator, -x)
        sum_val = sum_val + term
        
    # Handle the case where x <= 0
    result = tl.where(x_is_positive, sum_val, 0.0)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)


def zeta(input, other, *, out=None):
    # Ensure inputs are tensors
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    if not torch.is_tensor(other):
        other = torch.tensor(other)
    
    # Handle broadcasting
    shape = torch.broadcast_shapes(input.shape, other.shape)
    input = input.expand(shape)
    other = other.expand(shape)
    
    # Create output tensor
    if out is None:
        out = torch.empty(shape, dtype=torch.float32, device=input.device)
    else:
        if out.shape != shape:
            raise ValueError(f"Output shape {out.shape} does not match broadcast shape {shape}")
        
    # Get total number of elements
    n = input.numel()
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _zeta_kernel[grid](input, other, out, n, BLOCK=block)
    
    return out