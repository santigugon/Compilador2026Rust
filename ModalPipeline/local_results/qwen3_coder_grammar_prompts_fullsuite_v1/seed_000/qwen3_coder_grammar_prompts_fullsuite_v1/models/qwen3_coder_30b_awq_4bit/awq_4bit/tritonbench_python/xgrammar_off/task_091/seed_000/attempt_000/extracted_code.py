import torch
import triton
import triton.language as tl

@triton.jit
def erfc_sqrt_kernel(
    input_ptr,
    output1_ptr,
    output2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute erfc and sqrt
    output1 = 1.0 - tl.erf(input)
    output2 = tl.sqrt(input)
    
    tl.store(output1_ptr + offsets, output1, mask=mask)
    tl.store(output2_ptr + offsets, output2, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output1 = torch.empty_like(input)
    output2 = torch.empty_like(input)
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    erfc_sqrt_kernel[grid](
        input,
        output1,
        output2,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (output1, output2)
