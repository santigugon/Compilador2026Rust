import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def div_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    rounding_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    
    # Perform division
    result = input / other
    
    # Apply rounding if specified
    if rounding_mode == "floor":
        result = tl.floor(result)
    elif rounding_mode == "ceil":
        result = tl.ceil(result)
    elif rounding_mode == "round":
        result = tl.round(result)
    elif rounding_mode == "trunc":
        result = tl.trunc(result)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def div(input: torch.Tensor, other: Union[torch.Tensor, float, int], *, rounding_mode: Optional[str] = None, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Handle scalar other
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure both tensors are on the same device
    if other.device != input.device:
        other = other.to(input.device)
    
    # Handle broadcasting
    if input.shape != other.shape:
        # Use torch's broadcasting rules
        input, other = torch.broadcast_tensors(input, other)
    
    # Ensure output tensor is created if needed
    if out is None:
        out = torch.empty_like(input)
    
    # Determine the number of elements
    n_elements = input.numel()
    
    # Set up the kernel launch parameters
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    div_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        rounding_mode=rounding_mode,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
