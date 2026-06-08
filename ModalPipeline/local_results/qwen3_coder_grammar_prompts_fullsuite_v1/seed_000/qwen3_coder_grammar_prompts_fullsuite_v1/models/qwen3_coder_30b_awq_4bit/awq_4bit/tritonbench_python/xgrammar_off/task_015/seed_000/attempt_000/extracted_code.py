import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    input_ptr, other_ptr, output_ptr,
    input_size, other_size,
    alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < input_size
    
    input_ptrs = input_ptr + offsets
    other_ptrs = other_ptr + offsets
    
    input_vals = tl.load(input_ptrs, mask=mask)
    other_vals = tl.load(other_ptrs, mask=mask)
    
    output_vals = input_vals + alpha * other_vals
    
    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, output_vals, mask=mask)

def add(input, other, *, alpha=1, out=None):
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle broadcasting
    if input.shape != other.shape:
        # For simplicity, assuming broadcastable shapes
        # In practice, you'd want to handle this more carefully
        pass
    
    # Ensure tensors are on the same device and have compatible dtypes
    if other.device != input.device:
        other = other.to(input.device)
    
    # Determine output size
    output_size = max(input.numel(), other.numel())
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor must have the same shape as input tensor")
    
    # Launch kernel
    if output_size > 0:
        block_size = 256
        num_blocks = (output_size + block_size - 1) // block_size
        
        # Prepare pointers
        input_ptr = input.data_ptr()
        other_ptr = other.data_ptr()
        output_ptr = out.data_ptr()
        
        # Launch kernel
        add_kernel[
            num_blocks,
            1,
            (block_size,)
        ](
            input_ptr,
            other_ptr,
            output_ptr,
            input.numel(),
            other.numel(),
            alpha
        )
    
    return out
