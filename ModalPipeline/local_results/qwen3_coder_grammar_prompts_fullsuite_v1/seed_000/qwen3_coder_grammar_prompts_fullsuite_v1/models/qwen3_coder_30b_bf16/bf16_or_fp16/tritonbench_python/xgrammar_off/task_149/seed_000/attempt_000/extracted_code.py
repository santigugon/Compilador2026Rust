import torch
import triton
import triton.language as tl

@triton.jit
def hstack_div_kernel(
    input_ptr, divisor_ptr, output_ptr,
    input_size, divisor_size,
    rounding_mode,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size
    
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    divisor_vals = tl.load(divisor_ptr + offsets, mask=mask)
    
    result = input_vals / divisor_vals
    
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
    
    # Prepare divisor
    if not isinstance(divisor, torch.Tensor):
        divisor = torch.tensor(divisor, dtype=stacked.dtype, device=stacked.device)
    
    # Broadcast divisor to match stacked tensor shape
    if divisor.shape != stacked.shape:
        divisor = divisor.expand_as(stacked)
    
    # Determine rounding mode
    rounding_mode_code = 0
    if rounding_mode == 'trunc':
        rounding_mode_code = 1
    elif rounding_mode == 'floor':
        rounding_mode_code = 2
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(stacked)
    else:
        if out.shape != stacked.shape:
            raise ValueError("out tensor must have the same shape as the stacked tensor")
    
    # Launch kernel
    input_size = stacked.numel()
    divisor_size = divisor.numel()
    
    BLOCK_SIZE = 1024
    grid_size = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    hstack_div_kernel[grid_size](
        stacked.data_ptr(),
        divisor.data_ptr(),
        out.data_ptr(),
        input_size,
        divisor_size,
        rounding_mode_code,
        BLOCK_SIZE
    )
    
    return out
