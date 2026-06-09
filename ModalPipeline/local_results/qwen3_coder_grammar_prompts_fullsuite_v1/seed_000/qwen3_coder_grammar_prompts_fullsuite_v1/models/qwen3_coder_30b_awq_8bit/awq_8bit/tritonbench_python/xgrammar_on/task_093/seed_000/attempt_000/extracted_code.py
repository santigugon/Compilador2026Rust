import torch
import triton
import triton.language as tl

def softmax_log(input, dim=-1, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Handle negative dimension
    if dim < 0:
        dim = input.dim() + dim
    
    # Get input shape and compute total elements
    shape = input.shape
    total_elements = input.numel()
    
    # Compute the size of the softmax dimension
    softmax_dim_size = shape[dim]
    
    # Compute the product of all dimensions except the softmax dimension
    other_dims_size = 1
    for i, s in enumerate(shape):
        if i != dim:
            other_dims_size *= s
    
    # Create output tensor
    output = torch.empty_like(input)
    
    # Define block size
    BLOCK = 256
    
    # Helper kernel for log operation
    @triton.jit
    def _log_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.log(x)
        tl.store(out_ptr + offsets, y, mask=mask)
    
    # Helper kernel for softmax
    @triton.jit
    def _softmax_kernel(x_ptr, out_ptr, other_dims_size: tl.constexpr, softmax_dim_size: tl.constexpr, BLOCK: tl.constexpr):
        # Each block handles one element in the other dimensions
        pid = tl.program_id(0)
        
        # For each element in the other dimensions
        for i in range(other_dims_size):
            # Compute the starting offset for this element
            base_offset = i * softmax_dim_size
            
            # Load the data for this softmax dimension
            offsets = base_offset + tl.arange(0, softmax_dim_size)
            
            # Load data
            x = tl.load(x_ptr + offsets, mask=offsets < (i + 1) * softmax_dim_size, other=-float('inf'))
            
            # Compute max for numerical stability
            max_val = tl.max(x, axis=0)
            
            # Compute exp(x - max_val)
            x_shifted = x - max_val
            exp_x = tl.exp(x_shifted)
            
            # Compute sum of exp
            sum_exp = tl.sum(exp_x, axis=0)
            
            # Compute softmax
            softmax_x = exp_x / sum_exp
            
            # Store result
            tl.store(out_ptr + offsets, softmax_x, mask=offsets < (i + 1) * softmax_dim_size)
    
    # Apply log operation
    log_output = torch.empty_like(input)
    grid_log = (triton.cdiv(total_elements, BLOCK),)
    _log_kernel[grid_log](input, log_output, total_elements, BLOCK=BLOCK)
    
    # Apply softmax operation
    grid_softmax = (other_dims_size,)
    _softmax_kernel[grid_softmax](log_output, output, other_dims_size, softmax_dim_size, BLOCK=BLOCK)
    
    return output