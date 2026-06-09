import torch
import triton
import triton.language as tl

@triton.jit
def div_kernel(
    input_ptr, other_ptr, output_ptr,
    input_size, other_size,
    rounding_mode,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Handle broadcasting
    input_mask = offsets < input_size
    other_mask = offsets < other_size
    
    # Load data
    input_vals = tl.load(input_ptr + offsets, mask=input_mask)
    other_vals = tl.load(other_ptr + offsets, mask=other_mask)
    
    # Perform division
    result = input_vals / other_vals
    
    # Apply rounding if specified
    if rounding_mode == 0:  # "trunc"
        result = tl.where(result >= 0, tl.floor(result), tl.ceil(result))
    elif rounding_mode == 1:  # "floor"
        result = tl.floor(result)
    elif rounding_mode == 2:  # "round"
        result = tl.round(result)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=input_mask)

def div(input, other, *, rounding_mode=None, out=None):
    # Handle scalar inputs
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle type promotion
    if input.dtype != other.dtype:
        common_dtype = torch.result_type(input, other)
        input = input.to(common_dtype)
        other = other.to(common_dtype)
    
    # Determine rounding mode
    rounding_mode_int = -1
    if rounding_mode is not None:
        rounding_modes = {"trunc": 0, "floor": 1, "round": 2}
        if rounding_mode not in rounding_modes:
            raise ValueError(f"Unsupported rounding_mode: {rounding_mode}")
        rounding_mode_int = rounding_modes[rounding_mode]
    
    # Handle broadcasting
    input_size = input.numel()
    other_size = other.numel()
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor shape must match input tensor shape")
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid_size = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure tensors are contiguous
    input = input.contiguous()
    other = other.contiguous()
    out = out.contiguous()
    
    div_kernel[grid_size](
        input.data_ptr(),
        other.data_ptr(),
        out.data_ptr(),
        input_size,
        other_size,
        rounding_mode_int,
        BLOCK_SIZE
    )
    
    return out
