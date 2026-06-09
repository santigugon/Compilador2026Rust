import torch
import triton
import triton.language as tl

@triton.jit
def add_mean_kernel(
    input_ptr, other_ptr, output_ptr,
    input_size, other_size, output_size,
    alpha, dim_size, keepdim,
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
    
    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask)
    other_vals = tl.load(other_ptr + other_offsets, mask=other_mask)
    
    result = input_vals + alpha * other_vals
    tl.store(output_ptr + output_offsets, result, mask=output_mask)

@triton.jit
def mean_kernel(
    input_ptr, output_ptr,
    input_size, output_size,
    dim_size, keepdim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    input_offsets = offset + tl.arange(0, BLOCK_SIZE)
    output_offsets = offset + tl.arange(0, BLOCK_SIZE)
    
    input_mask = input_offsets < input_size
    output_mask = output_offsets < output_size
    
    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask)
    mean_val = tl.sum(input_vals) / dim_size
    tl.store(output_ptr + output_offsets, mean_val, mask=output_mask)

def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)
        if isinstance(other, torch.Tensor):
            other = other.to(dtype)
    
    if isinstance(other, (int, float, complex)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcasting
    if input.shape != other.shape:
        input, other = torch.broadcast_tensors(input, other)
    
    # Add operation
    result = input + alpha * other
    
    # Mean computation
    if dim is None:
        # Compute mean over all elements
        output = torch.mean(result)
    else:
        output = torch.mean(result, dim=dim, keepdim=keepdim)
    
    if out is not None:
        out.copy_(output)
        return out
    
    return output
