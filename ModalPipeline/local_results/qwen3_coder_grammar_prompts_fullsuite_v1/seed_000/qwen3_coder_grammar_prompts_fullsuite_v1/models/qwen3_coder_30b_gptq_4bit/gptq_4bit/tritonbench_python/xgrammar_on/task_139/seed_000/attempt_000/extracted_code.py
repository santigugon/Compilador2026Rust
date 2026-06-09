import torch
import triton
import triton.language as tl

def _get_reduction_dims(input, dim):
    if dim is None:
        return list(range(input.ndim))
    if isinstance(dim, int):
        return [dim]
    return list(dim)

def _get_output_shape(input, dim, keepdim):
    reduction_dims = _get_reduction_dims(input, dim)
    output_shape = []
    for i in range(input.ndim):
        if i not in reduction_dims:
            output_shape.append(input.shape[i])
        elif keepdim:
            output_shape.append(1)
    return output_shape

def _get_numel(input, dim, keepdim):
    reduction_dims = _get_reduction_dims(input, dim)
    numel = 1
    for i in range(input.ndim):
        if i not in reduction_dims:
            numel *= input.shape[i]
        elif keepdim:
            numel *= 1
    return numel

@triton.jit
def _std_kernel(
    input_ptr,
    output_ptr,
    n,
    correction,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Compute mean
    mean = tl.sum(x) / n
    # Compute variance
    diff = x - mean
    squared_diff = diff * diff
    variance = tl.sum(squared_diff) / (n - correction)
    # Compute standard deviation
    std = tl.sqrt(variance)
    tl.store(output_ptr + offsets, std, mask=mask)

def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    # Handle scalar input
    if input.ndim == 0:
        if out is not None:
            out.fill_(0.0)
        return torch.tensor(0.0)
    
    # Handle the case where dim is None (reduce all dimensions)
    if dim is None:
        # Flatten the tensor
        flat_input = input.flatten()
        n = flat_input.numel()
        if out is not None:
            out = torch.empty((), dtype=input.dtype, device=input.device)
        else:
            out = torch.empty((), dtype=input.dtype, device=input.device)
        
        # Use PyTorch for the actual computation when reducing all dimensions
        # This is because we need to handle the case where we reduce all dimensions
        # and the output is a scalar
        result = torch.std(input, correction=correction)
        if out is not None:
            out.copy_(result)
            return out
        return result
    
    # For non-None dim, we need to compute the std along specific dimensions
    # Get the output shape
    output_shape = _get_output_shape(input, dim, keepdim)
    
    # Create output tensor
    if out is not None:
        if out.shape != tuple(output_shape):
            raise ValueError("Output tensor shape does not match expected shape")
        out = out
    else:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Get the reduction dimensions
    reduction_dims = _get_reduction_dims(input, dim)
    
    # Calculate the number of elements in the output
    output_numel = 1
    for i in range(len(output_shape)):
        output_numel *= output_shape[i]
    
    # For each reduction dimension, we need to compute the std
    # We'll compute it by flattening the input along the reduction dimensions
    # and then compute the std for each element in the output
    
    # Get the number of elements in the flattened tensor
    flat_input = input
    for dim in sorted(reduction_dims, reverse=True):
        flat_input = flat_input.flatten(start_dim=dim, end_dim=dim)
    
    # Get the number of elements in the flattened tensor
    n = flat_input.numel()
    
    # If we have a single element, return 0.0
    if n <= 1:
        out.fill_(0.0)
        return out
    
    # For the case where we reduce along a single dimension
    # We'll compute the std for each element in the output
    block = 256
    grid = (triton.cdiv(output_numel, block),)
    
    # Create a temporary tensor for the computation
    temp_out = torch.empty(output_numel, dtype=input.dtype, device=input.device)
    
    # Compute the std
    _std_kernel[grid](input, temp_out, n, correction, BLOCK=block)
    
    # Reshape the output
    out = temp_out.reshape(output_shape)
    
    return out