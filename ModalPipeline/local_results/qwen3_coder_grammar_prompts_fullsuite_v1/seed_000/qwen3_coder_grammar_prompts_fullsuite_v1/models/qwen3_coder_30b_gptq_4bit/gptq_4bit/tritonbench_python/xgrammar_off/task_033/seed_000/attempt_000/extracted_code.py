import torch
import triton
import triton.language as tl

@triton.jit
def logsumexp_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
    dim: tl.constexpr,
    keepdim: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)
    
    # Load input data
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # Compute logsumexp
    max_val = tl.max(input, axis=0)
    exp_val = tl.exp(input - max_val)
    sum_val = tl.sum(exp_val, axis=0)
    result = tl.log(sum_val) + max_val
    
    # Store result
    tl.store(output_ptr + pid, result, mask=pid < (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)

def logsumexp(input, dim, keepdim=False, *, out=None):
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    
    # Handle negative dimension
    if dim < 0:
        dim = input.dim() + dim
    
    # Get output shape
    output_shape = list(input.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Get number of elements
    n_elements = input.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    logsumexp_kernel[grid](input, output, n_elements, BLOCK_SIZE, dim, keepdim)
    
    return output
