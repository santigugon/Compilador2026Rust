import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    input_ptr, other_ptr, output_ptr,
    input_size, other_size, output_size,
    dtype_input, dtype_other, dtype_output,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    input_offsets = offset + tl.arange(0, BLOCK_SIZE)
    other_offsets = offset + tl.arange(0, BLOCK_SIZE)
    output_offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    input_mask = input_offsets < input_size
    other_mask = other_offsets < other_size
    output_mask = output_offsets < output_size
    
    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0)
    other_vals = tl.load(other_ptr + other_offsets, mask=other_mask, other=0)
    
    # Element-wise multiplication
    result = input_vals * other_vals
    
    tl.store(output_ptr + output_offsets, result, mask=output_mask)

def mul(input, other, *, out=None):
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other)
    
    # Determine output shape through broadcasting
    output_shape = torch.broadcast_shapes(input.shape, other.shape)
    
    # Determine output dtype through type promotion
    output_dtype = torch.result_type(input, other)
    
    # Create output tensor
    if out is not None:
        if out.shape != output_shape or out.dtype != output_dtype:
            raise ValueError("Output tensor shape or dtype does not match expected values")
        output = out
    else:
        output = torch.empty(output_shape, dtype=output_dtype, device=input.device)
    
    # Handle device placement
    if input.device != other.device:
        other = other.to(input.device)
    
    # Flatten tensors for kernel execution
    input_flat = input.flatten()
    other_flat = other.flatten()
    output_flat = output.flatten()
    
    # Launch kernel
    if input_flat.numel() > 0 and other_flat.numel() > 0:
        BLOCK_SIZE = 1024
        grid_size = (output_flat.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        mul_kernel[grid_size](
            input_flat.data_ptr(), other_flat.data_ptr(), output_flat.data_ptr(),
            input_flat.numel(), other_flat.numel(), output_flat.numel(),
            input.dtype, other.dtype, output.dtype,
            BLOCK_SIZE
        )
    
    return output
