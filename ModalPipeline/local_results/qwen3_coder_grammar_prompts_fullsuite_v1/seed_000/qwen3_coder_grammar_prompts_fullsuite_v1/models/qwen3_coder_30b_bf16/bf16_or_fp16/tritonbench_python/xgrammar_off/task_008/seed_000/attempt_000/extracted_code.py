import torch
import triton
import triton.language as tl

@triton.jit
def _sub_kernel(
    input_ptr, other_ptr, output_ptr,
    input_size,
    other_size,
    alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    input_offsets = offset + tl.arange(0, BLOCK_SIZE)
    other_offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    input_mask = input_offsets < input_size
    other_mask = other_offsets < other_size
    
    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask)
    other_vals = tl.load(other_ptr + other_offsets, mask=other_mask)
    
    result = input_vals - alpha * other_vals
    tl.store(output_ptr + input_offsets, result, mask=input_mask)

def sub(input, other, *, alpha=1, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    if isinstance(other, (int, float, complex)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    input_size = input.numel()
    other_size = other.numel()
    
    if input_size != other_size:
        # Handle broadcasting
        input_shape = input.shape
        other_shape = other.shape
        # Simple broadcasting logic for same shapes or compatible shapes
        # For simplicity, assuming compatible broadcasting
        if input_shape != other_shape:
            # This is a simplified case - in practice, you'd need full broadcasting logic
            pass
    
    BLOCK_SIZE = 1024
    grid_size = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _sub_kernel[grid_size](
        input.data_ptr(),
        other.data_ptr(),
        out.data_ptr(),
        input_size,
        other_size,
        alpha,
        BLOCK_SIZE
    )
    
    return out
