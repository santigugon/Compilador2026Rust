import torch
import triton
import triton.language as tl

@triton.jit
def _std_kernel(
    input_ptr, 
    output_ptr,
    n_elements,
    correction,
    keepdim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    # Calculate mean
    mean = tl.sum(input) / n_elements
    
    # Calculate variance
    diff = input - mean
    squared_diff = diff * diff
    variance = tl.sum(squared_diff) / (n_elements - correction)
    
    # Calculate std
    std = tl.sqrt(variance)
    
    tl.store(output_ptr + pid, std, mask=mask)

def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    if dim is None:
        # Reduce over all dimensions
        input_flat = input.flatten()
        n_elements = input_flat.numel()
        output = torch.empty((), dtype=input.dtype, device=input.device)
        if n_elements > 0:
            grid = (triton.cdiv(n_elements, 1024),)
            _std_kernel[grid](
                input_flat.data_ptr(),
                output.data_ptr(),
                n_elements,
                correction,
                keepdim,
                BLOCK_SIZE=1024
            )
        else:
            output.fill_(0)
        if keepdim:
            output = output.reshape([1] * input.dim())
        return output
    else:
        # Reduce over specified dimensions
        if isinstance(dim, int):
            dim = [dim]
        # Normalize negative dimensions
        dim = [d if d >= 0 else input.dim() + d for d in dim]
        # Sort dimensions in descending order to avoid index shifting issues
        dim = sorted(dim, reverse=True)
        
        # Create output shape
        output_shape = list(input.shape)
        for d in dim:
            output_shape[d] = 1 if keepdim else 1
        
        # Flatten input and output for easier processing
        input_flat = input
        for d in dim:
            input_flat = input_flat.sum(dim=d, keepdim=True)
        
        # Calculate std for each element
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        n_elements = input_flat.numel()
        if n_elements > 0:
            grid = (triton.cdiv(n_elements, 1024),)
            _std_kernel[grid](
                input_flat.data_ptr(),
                output.data_ptr(),
                n_elements,
                correction,
                keepdim,
                BLOCK_SIZE=1024
            )
        else:
            output.fill_(0)
        return output
