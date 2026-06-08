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
    input_offsets = offsets % input_size
    other_offsets = offsets % other_size
    
    input_data = tl.load(input_ptr + input_offsets, mask=input_offsets < input_size)
    other_data = tl.load(other_ptr + other_offsets, mask=other_offsets < other_size)
    
    # Perform division
    result = input_data / other_data
    
    # Apply rounding mode if specified
    if rounding_mode == "trunc":
        result = tl.trunc(result)
    elif rounding_mode == "floor":
        result = tl.floor(result)
    elif rounding_mode == "round":
        result = tl.round(result)
    
    tl.store(output_ptr + offsets, result)

def div(input, other, *, rounding_mode=None, out=None):
    # Convert inputs to tensors if they are not already
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other)
    
    # Ensure inputs are on the same device and have compatible dtypes
    if input.device != other.device:
        other = other.to(input.device)
    
    # Handle type promotion
    if input.dtype != other.dtype:
        common_dtype = torch.promote_types(input.dtype, other.dtype)
        input = input.to(common_dtype)
        other = other.to(common_dtype)
    
    # Determine output size and shape
    output_shape = torch.broadcast_shapes(input.shape, other.shape)
    output_size = output_shape.numel()
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        if out.shape != output_shape or out.dtype != input.dtype or out.device != input.device:
            raise ValueError("Output tensor must have the same shape, dtype, and device as the result")
    
    # Launch kernel
    if output_size > 0:
        BLOCK_SIZE = 1024
        num_blocks = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (num_blocks,)
        
        # Convert to appropriate types for Triton
        input_ptr = input.data_ptr()
        other_ptr = other.data_ptr()
        output_ptr = out.data_ptr()
        
        # Determine rounding mode string
        rounding_mode_str = ""
        if rounding_mode is not None:
            rounding_mode_str = rounding_mode
        
        div_kernel[grid](
            input_ptr, other_ptr, output_ptr,
            input.numel(), other.numel(),
            rounding_mode_str,
            BLOCK_SIZE
        )
    
    return out
