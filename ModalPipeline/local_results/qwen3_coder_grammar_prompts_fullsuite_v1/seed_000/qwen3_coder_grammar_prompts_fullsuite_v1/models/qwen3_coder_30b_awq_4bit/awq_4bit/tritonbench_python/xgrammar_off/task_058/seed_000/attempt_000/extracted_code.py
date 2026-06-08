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
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    if eps is not None:
        input = tl.clamp(input, eps, 1.0 - eps)
    
    output = tl.log(input / (1.0 - input))
    tl.store(output_ptr + offsets, output, mask=mask)

def logit(input, eps=None, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32, device=input.device)
    
    if input.numel() == 0:
        return out
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    if eps is not None:
        eps = eps if isinstance(eps, float) else float(eps)
        logit_kernel[grid](
            input_ptr=input,
            output_ptr=out,
            eps=eps,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        logit_kernel[grid](
            input_ptr=input,
            output_ptr=out,
            eps=None,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return out
