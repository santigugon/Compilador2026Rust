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
    
    input_data = tl.load(input_ptr + input_offsets, mask=offsets < input_size)
    other_data = tl.load(other_ptr + other_offsets, mask=offsets < other_size)
    
    # Perform division
    result = input_data / other_data
    
    # Apply rounding mode if specified
    if rounding_mode == "trunc":
        result = tl.trunc(result)
    elif rounding_mode == "floor":
        result = tl.floor(result)
    elif rounding_mode == "round":
        result = tl.round(result)
    
    tl.store(output_ptr + offsets, result, mask=offsets < input_size)

def div(input, other, *, rounding_mode=None, out=None):
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other)
    
    # Handle broadcasting
    input_size = input.numel()
    other_size = other.numel()
    
    # Determine output size
    if input_size >= other_size:
        output_size = input_size
        input_broadcast = False
        other_broadcast = True
    else:
        output_size = other_size
        input_broadcast = True
        other_broadcast = False
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    
    # Ensure tensors are on the same device and have compatible dtypes
    if input.device != other.device:
        other = other.to(input.device)
    
    # Determine the output dtype
    if input.dtype == torch.complex64 or other.dtype == torch.complex64:
        out_dtype = torch.complex64
    elif input.dtype == torch.complex128 or other.dtype == torch.complex128:
        out_dtype = torch.complex128
    elif input.dtype == torch.float32 or other.dtype == torch.float32:
        out_dtype = torch.float32
    elif input.dtype == torch.float64 or other.dtype == torch.float64:
        out_dtype = torch.float64
    else:
        out_dtype = torch.float32
    
    out = out.to(out_dtype)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid_size = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    other_ptr = other.data_ptr()
    output_ptr = out.data_ptr()
    
    # Launch kernel
    div_kernel[grid_size](
        input_ptr, other_ptr, output_ptr,
        input_size, other_size,
        rounding_mode,
        BLOCK_SIZE
    )
    
    return out
