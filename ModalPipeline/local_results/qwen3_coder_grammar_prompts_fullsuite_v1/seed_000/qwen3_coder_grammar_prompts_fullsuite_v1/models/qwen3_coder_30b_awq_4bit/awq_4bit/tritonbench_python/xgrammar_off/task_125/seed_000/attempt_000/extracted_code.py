import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    output = 1.0 / (1.0 + tl.exp(-input))
    tl.store(output_ptr + offsets, output, mask=mask)

def sigmoid(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, 256),)
    sigmoid_kernel[grid](
        input,
        out,
        n_elements,
        BLOCK_SIZE=256,
    )
    return out
