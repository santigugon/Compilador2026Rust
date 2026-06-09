import torch
import triton
import triton.language as tl

@triton.jit
def div_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    rounding_mode,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    
    # Perform division
    output = input / other
    
    # Apply rounding if specified
    if rounding_mode == "floor":
        output = tl.floor(output)
    elif rounding_mode == "ceil":
        output = tl.ceil(output)
    elif rounding_mode == "trunc":
        output = tl.trunc(output)
    elif rounding_mode == "round":
        output = tl.round(output)
    
    tl.store(output_ptr + offsets, output, mask=mask)

def div(input, other, *, rounding_mode=None, out=None):
    if isinstance(other, (int, float, complex)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        other = other.to(input.dtype).to(input.device)
    
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure tensors have the same device
    if input.device != other.device:
        other = other.to(input.device)
    
    # Handle broadcasting
    input, other = torch.broadcast_tensors(input, other)
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Launch kernel
    grid = (triton.cdiv(n_elements, 1024),)
    div_kernel[grid](
        input_ptr=input.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        rounding_mode=rounding_mode,
        BLOCK_SIZE=1024
    )
    
    return out
