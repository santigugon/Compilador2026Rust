import torch
import triton
import triton.language as tl

@triton.jit
def logit_kernel(
    input_ptr,
    output_ptr,
    eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # Clamp input to [eps, 1 - eps]
    if eps != 0.0:
        input = tl.clamp(input, eps, 1.0 - eps)
    
    # Compute logit
    output = tl.math.log(input / (1.0 - input))
    
    tl.store(output_ptr + offsets, output, mask=mask)

def logit(input, eps=None, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    if eps is None:
        eps = 0.0
    else:
        eps = float(eps)
    
    # Ensure input is in valid range for clamping
    if eps != 0.0:
        input = torch.clamp(input, eps, 1.0 - eps)
    
    # Use Triton kernel
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    logit_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=out.data_ptr(),
        eps=eps,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
