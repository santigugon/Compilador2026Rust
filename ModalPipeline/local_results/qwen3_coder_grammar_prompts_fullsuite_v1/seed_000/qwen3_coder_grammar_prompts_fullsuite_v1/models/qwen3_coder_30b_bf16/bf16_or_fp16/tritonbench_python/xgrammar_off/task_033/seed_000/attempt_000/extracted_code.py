import torch
import triton
import triton.language as tl

@triton.jit
def logsumexp_kernel(
    input_ptr, 
    output_ptr, 
    n_elements,
    dim_size,
    stride_input_dim,
    stride_output_dim,
    keepdim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    
    # Find max value for numerical stability
    max_val = tl.max(input_vals)
    
    # Compute sum of exponentials
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals)
    
    # Compute log of sum
    result = tl.log(sum_exp) + max_val
    
    # Store result
    if keepdim:
        output_offsets = offsets // dim_size * stride_output_dim + (offsets % dim_size)
    else:
        output_offsets = offsets
    tl.store(output_ptr + output_offsets, result, mask=mask)

def logsumexp(input, dim, keepdim=False, *, out=None) -> torch.Tensor:
    # Validate input
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    
    if not input.is_contiguous():
        input = input.contiguous()
    
    # Handle negative dimensions
    if dim < 0:
        dim = input.dim() + dim
    
    # Validate dimension
    if dim < 0 or dim >= input.dim():
        raise IndexError("dim out of range")
    
    # Create output tensor
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.shape != input.shape:
            raise ValueError("out tensor must have the same shape as input")
    else:
        # Compute output shape
        output_shape = list(input.shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Handle empty tensor
    if input.numel() == 0:
        out.fill_(float('-inf'))
        return out
    
    # Get dimensions
    input_size = input.numel()
    dim_size = input.shape[dim]
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 1024
    grid_size = (input_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Compute strides
    stride_input_dim = 1
    stride_output_dim = 1
    for i in range(dim + 1, input.dim()):
        stride_input_dim *= input.shape[i]
    for i in range(len(out.shape)):
        stride_output_dim *= out.shape[i] if i < len(out.shape) - 1 else 1
    
    # Launch kernel
    logsumexp_kernel[grid_size](
        input.data_ptr(),
        out.data_ptr(),
        input_size,
        dim_size,
        stride_input_dim,
        stride_output_dim,
        keepdim,
        BLOCK_SIZE
    )
    
    return out
