import torch
import triton
import triton.language as tl

@triton.jit
def i0_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute I0 using series expansion
    # I0(x) = 1 + sum_{n=1}^∞ (x^n / (2^n * n!))^2 / (n!)^2
    
    # Initialize with 1
    result = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    
    # Series expansion: compute terms until convergence
    x = input
    x_squared = x * x
    term = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
    n = tl.full([BLOCK_SIZE], 1, dtype=tl.int32)
    
    # We'll compute a few terms for reasonable accuracy
    for i in range(20):
        # Compute next term: (x^n / (2^n * n!))^2 / (n!)^2
        term = term * (x_squared / 4.0) / (n * n)
        result = result + term
        n = n + 1
    
    tl.store(output_ptr + offsets, result, mask=mask)

def i0(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    
    # Ensure input is float32 for computation
    input = input.float()
    
    # Launch kernel
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    i0_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
