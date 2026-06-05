import torch
from typing import Tuple
import triton
import triton.language as tl

@triton.jit
def erfc_sqrt_kernel(
    input_ptr,
    erfc_ptr,
    sqrt_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute erfc
    erfc = tl.math.erfc(input)
    tl.store(erfc_ptr + offsets, erfc, mask=mask)
    
    # Compute sqrt
    sqrt = tl.math.sqrt(input)
    tl.store(sqrt_ptr + offsets, sqrt, mask=mask)

def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.is_cuda, "Input tensor must be on CUDA device"
    n_elements = input.numel()
    output_shape = input.shape
    
    # Allocate output tensors
    erfc_output = torch.empty_like(input)
    sqrt_output = torch.empty_like(input)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    erfc_sqrt_kernel[grid](
        input_ptr=input.data_ptr(),
        erfc_ptr=erfc_output.data_ptr(),
        sqrt_ptr=sqrt_output.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return (erfc_output, sqrt_output)
