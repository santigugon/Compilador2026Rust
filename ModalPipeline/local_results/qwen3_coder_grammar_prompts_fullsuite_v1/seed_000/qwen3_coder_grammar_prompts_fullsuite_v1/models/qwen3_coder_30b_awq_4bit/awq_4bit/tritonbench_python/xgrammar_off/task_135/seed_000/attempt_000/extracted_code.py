import torch
import triton
import triton.language as tl

@triton.jit
def mul_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    output = input * other
    tl.store(output_ptr + offsets, output, mask=mask)

def mul(input, other, *, out=None):
    if not isinstance(input, torch.Tensor):
        input = torch.tensor(input)
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other)
    
    # Handle broadcasting
    if input.shape != other.shape:
        # For simplicity, we'll use torch's broadcasting
        # In a real implementation, we'd need to handle this properly
        pass
    
    # Ensure both tensors are on the same device and have compatible dtypes
    if input.device != other.device:
        other = other.to(input.device)
    
    # Determine output shape and dtype
    output_shape = torch.broadcast_tensors(input, other)[0].shape
    output_dtype = torch.result_type(input, other)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=output_dtype, device=input.device)
    else:
        if out.shape != output_shape or out.dtype != output_dtype:
            raise ValueError("Output tensor has incorrect shape or dtype")
    
    # Prepare for Triton kernel
    input = input.contiguous()
    other = other.contiguous()
    out = out.contiguous()
    
    # Launch kernel
    size = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(size, BLOCK_SIZE),)
    
    mul_kernel[grid](
        input.data_ptr(),
        other.data_ptr(),
        out.data_ptr(),
        size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
