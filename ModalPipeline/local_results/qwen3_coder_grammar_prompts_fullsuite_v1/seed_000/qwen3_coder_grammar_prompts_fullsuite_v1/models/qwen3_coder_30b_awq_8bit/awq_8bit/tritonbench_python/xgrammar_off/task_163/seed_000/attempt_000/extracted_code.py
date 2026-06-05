import torch
import triton
import triton.language as tl

@triton.jit
def cos_signbit_kernel(
    input_ptr,
    output_cos_ptr,
    output_signbit_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    cos_val = tl.cos(input)
    tl.store(output_cos_ptr + offsets, cos_val, mask=mask)
    signbit = tl.where(cos_val >= 0, 0, 1)
    tl.store(output_signbit_ptr + offsets, signbit, mask=mask)

def cos_signbit(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output_cos = torch.empty_like(input)
    output_signbit = torch.empty_like(input, dtype=torch.int8)
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    cos_signbit_kernel[grid](
        input,
        output_cos,
        output_signbit,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_cos, output_signbit
