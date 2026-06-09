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
    
    # Compute Hurwitz zeta function using the series representation
    # ζ(x,q) = Σ_{n=0}^∞ 1/(n+q)^x
    # We'll compute a finite sum for practical purposes
    # Using a reasonable number of terms (50) for convergence
    result = tl.zeros([BLOCK], dtype=tl.float32)
    term = tl.ones([BLOCK], dtype=tl.float32)
    
    # For x <= 1, we need special handling or more terms
    # Here we'll use a simple approach with fixed terms
    for i in range(50):
        n_val = i + 1
        denominator = tl.pow(n_val + q, x)
        # Avoid division by zero
        denominator = tl.where(denominator == 0, 1.0, denominator)
        term = 1.0 / denominator
        result = result + term
        
        # Early stopping if terms become negligible
        if tl.all(tl.abs(term) < 1e-10):
            break
    
    # Add the first term (n=0) which is 1/q^x
    first_term = 1.0 / tl.pow(q, x)
    result = result + first_term
    
    tl.store(out_ptr + offsets, result, mask=mask)

def zeta(input, other, *, out=None):
    # Handle scalar inputs
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=torch.float32)
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=torch.float32)
    
    # Ensure inputs are float32
    input = input.to(torch.float32)
    other = other.to(torch.float32)
    
    # Handle broadcasting
    shape = torch.broadcast_shapes(input.shape, other.shape)
    input = input.expand(shape)
    other = other.expand(shape)
    
    # Create output tensor
    if out is None:
        out = torch.empty(shape, dtype=torch.float32, device=input.device)
    else:
        if out.shape != shape or out.dtype != torch.float32 or out.device != input.device:
            raise ValueError("Output tensor has incorrect shape, dtype, or device")
    
    # Launch kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _zeta_kernel[grid](input, other, out, n, BLOCK=block)
    return out
