import torch
from typing import Tuple
import triton
import triton.language as tl

@triton.jit
def signbit_kernel(input_ptr, other_ptr, output1_ptr, output2_ptr, n_elements, BLOCK_SIZE: int = 1024):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    
    # Compute signbit
    signbit_result = tl.where(input < 0, 1, 0)
    signbit_result = tl.where((input == 0) & (tl.load(input_ptr + offsets) < 0), 1, signbit_result)
    tl.store(output1_ptr + offsets, signbit_result, mask=mask)
    
    # Compute bitwise_and
    bitwise_and_result = input & other
    tl.store(output2_ptr + offsets, bitwise_and_result, mask=mask)

def signbit_bitwise_and(input: torch.Tensor, other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input.dtype == torch.float32 or input.dtype == torch.float64, "input must be float32 or float64"
    assert other.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.bool], "other must be integral or boolean type"
    assert input.shape == other.shape, "input and other must have the same shape"
    
    n_elements = input.numel()
    output1 = torch.zeros_like(input, dtype=torch.bool)
    output2 = torch.empty_like(other)
    
    grid = (triton.cdiv(n_elements, 1024),)
    signbit_kernel[grid](
        input_ptr=input,
        other_ptr=other,
        output1_ptr=output1,
        output2_ptr=output2,
        n_elements=n_elements
    )
    
    return output1, output2
