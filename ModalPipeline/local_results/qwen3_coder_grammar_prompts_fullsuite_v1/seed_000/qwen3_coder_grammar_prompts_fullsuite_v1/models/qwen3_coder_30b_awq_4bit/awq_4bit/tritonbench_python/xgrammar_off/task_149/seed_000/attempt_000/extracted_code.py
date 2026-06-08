import torch
import triton
import triton.language as tl

@triton.jit
def hstack_div_kernel(
    input_ptr,
    divisor_ptr,
    output_ptr,
    num_elements,
    rounding_mode,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    divisor = tl.load(divisor_ptr + offsets, mask=mask)
    result = input / divisor
    
    if rounding_mode == 1:  # trunc
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == 2:  # floor
        result = tl.floor(result)
    
    tl.store(output_ptr + offsets, result, mask=mask)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    if not tensors:
        raise ValueError("tensors must not be empty")
    
    # Stack tensors horizontally
    stacked = torch.cat(tensors, dim=-1)
    
    # Handle divisor
    if isinstance(divisor, (int, float)):
        divisor = torch.tensor(divisor, dtype=stacked.dtype, device=stacked.device)
    else:
        divisor = divisor.to(stacked.dtype).to(stacked.device)
    
    # Broadcast divisor to match stacked shape
    if divisor.shape != stacked.shape:
        divisor = divisor.expand_as(stacked)
    
    # Prepare output
    if out is None:
        out = torch.empty_like(stacked)
    else:
        if out.shape != stacked.shape:
            raise ValueError("out tensor must have the same shape as the stacked tensor")
    
    # Determine rounding mode
    rounding_mode_code = 0
    if rounding_mode == 'trunc':
        rounding_mode_code = 1
    elif rounding_mode == 'floor':
        rounding_mode_code = 2
    
    # Launch kernel
    num_elements = stacked.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    hstack_div_kernel[grid](
        stacked.data_ptr(),
        divisor.data_ptr(),
        out.data_ptr(),
        num_elements,
        rounding_mode_code,
        BLOCK_SIZE
    )
    
    return out
