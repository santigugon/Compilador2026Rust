import torch
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(
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
    output = tl.signbit(input)
    tl.store(output_ptr + offsets, output, mask=mask)

def signbit(input, *, out=None):
    assert input.dtype == torch.float32 or input.dtype == torch.float64, "Only float32 and float64 are supported"
    
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)
    else:
        assert out.dtype == torch.bool, "Output tensor must be of boolean type"
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    signbit_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
