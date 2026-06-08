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
    result = 0.0
    for i in range(50):
        n_val = i + q
        # Avoid division by zero and handle negative exponents
        if n_val > 0:
            result += 1.0 / (n_val ** x)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def zeta(input, other, *, out=None):
    # Handle scalar inputs
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=torch.float32, device=other.device)
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=torch.float32, device=input.device)
    
    # Ensure inputs are the same shape for broadcasting
    input, other = torch.broadcast_tensors(input, other)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensors")
    
    # Launch kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _zeta_kernel[grid](input, other, out, n, BLOCK=block)
    return out
