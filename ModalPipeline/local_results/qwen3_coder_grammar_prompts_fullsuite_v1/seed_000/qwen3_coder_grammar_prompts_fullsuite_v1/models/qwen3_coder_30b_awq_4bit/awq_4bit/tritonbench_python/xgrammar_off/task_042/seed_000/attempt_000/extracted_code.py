import torch
import triton
import triton.language as tl

@triton.jit
def zeta_kernel(x_ptr, q_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    q = tl.load(q_ptr + offsets, mask=mask)
    
    # Initialize result
    result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute Hurwitz zeta function using series approximation
    # zeta(x, q) = sum_{n=0}^{inf} (q + n)^{-x}
    # We'll compute a finite sum for practical purposes
    max_iter = 1000
    for i in range(max_iter):
        n = i
        term = (q + n) ** (-x)
        result = tl.where(tl.abs(term) > 1e-12, result + term, result)
        
    tl.store(out_ptr + offsets, result, mask=mask)

def zeta(input, other, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    
    # Ensure inputs are float32
    input = input.to(torch.float32)
    other = other.to(torch.float32)
    out = out.to(torch.float32)
    
    # Launch kernel
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    zeta_kernel[grid](
        input,
        other,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
