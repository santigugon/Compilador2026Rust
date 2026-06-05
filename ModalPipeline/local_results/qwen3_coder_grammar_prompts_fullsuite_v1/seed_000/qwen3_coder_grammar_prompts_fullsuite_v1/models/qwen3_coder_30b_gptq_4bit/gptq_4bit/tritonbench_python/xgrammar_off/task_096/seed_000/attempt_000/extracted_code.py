import torch
import triton
import triton.language as tl

@triton.jit
def gammaln_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = tl.libdevice.tlgammaln(input)
    tl.store(output_ptr + offsets, output, mask=mask)

def gammaln(input, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    else:
        assert out.dtype == torch.float32, "Output tensor must be of type float32"
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    gammaln_kernel[grid](
        input_ptr=input,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
