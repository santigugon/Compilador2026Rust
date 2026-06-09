import torch
import triton
import triton.language as tl

def _get_reduction_dims(input_shape, dim):
    if dim is None:
        return list(range(len(input_shape)))
    if isinstance(dim, int):
        dim = [dim]
    # Handle negative dimensions
    dim = [d if d >= 0 else d + len(input_shape) for d in dim]
    return sorted(dim)

def _get_output_shape(input_shape, reduction_dims, keepdim):
    if keepdim:
        return [1 if i in reduction_dims else input_shape[i] for i in range(len(input_shape))]
    else:
        return [input_shape[i] for i in range(len(input_shape)) if i not in reduction_dims]

def _get_numel(shape):
    result = 1
    for s in shape:
        result *= s
    return result

@triton.jit
def _mean_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute sum
    sum_val = tl.sum(x, axis=0)
    # Compute mean
    mean_val = sum_val / n
    tl.store(out_ptr, mean_val, mask=mask)

@triton.jit
def _var_kernel(x_ptr, mean_ptr, out_ptr, n: tl.constexpr, correction: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr)
    # Compute squared differences
    diff = x - mean
    squared_diff = diff * diff
    # Compute sum of squared differences
    sum_squared_diff = tl.sum(squared_diff, axis=0)
    # Compute variance
    var_val = sum_squared_diff / (n - correction)
    tl.store(out_ptr, var_val, mask=mask)

@triton.jit
def _std_kernel(x_ptr, mean_ptr, out_ptr, n: tl.constexpr, correction: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr)
    # Compute squared differences
    diff = x - mean
    squared_diff = diff * diff
    # Compute sum of squared differences
    sum_squared_diff = tl.sum(squared_diff, axis=0)
    # Compute variance
    var_val = sum_squared_diff / (n - correction)
    # Compute standard deviation
    std_val = tl.sqrt(var_val)
    tl.store(out_ptr, std_val, mask=mask)

def std(input, dim=None, *, correction=1, keepdim=False, out=None):
    # Handle scalar input
    if input.numel() == 1:
        if out is not None:
            out.fill_(0.0)
            return out
        return torch.zeros_like(input)
    
    # Get reduction dimensions
    reduction_dims = _get_reduction_dims(input.shape, dim)
    
    # Calculate output shape
    output_shape = _get_output_shape(input.shape, reduction_dims, keepdim)
    
    # Calculate number of elements to reduce
    numel_to_reduce = 1
    for i in reduction_dims:
        numel_to_reduce *= input.shape[i]
    
    # Calculate total number of elements in output
    output_numel = _get_numel(output_shape)
    
    # If output is provided, use it
    if out is not None:
        result = out
    else:
        result = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # If no reduction needed, return input
    if numel_to_reduce == 1:
        if out is not None:
            out.copy_(input)
            return out
        return input.clone()
    
    # For single dimension reduction
    if len(reduction_dims) == 1:
        # Flatten input to 1D for reduction
        flat_input = input.view(-1)
        
        # Compute mean
        mean = flat_input.sum() / numel_to_reduce
        
        # Compute variance
        diff = flat_input - mean
        squared_diff = diff * diff
        sum_squared_diff = squared_diff.sum()
        var_val = sum_squared_diff / (numel_to_reduce - correction)
        
        # Compute standard deviation
        std_val = torch.sqrt(var_val)
        
        # Reshape result
        if keepdim:
            result = std_val.view(output_shape)
        else:
            result = std_val
        
        if out is not None:
            out.copy_(result)
            return out
        return result
    
    # For multi-dimension reduction, use a more complex approach
    # This is a simplified version that handles the case where we reduce over all dimensions
    if len(reduction_dims) == len(input.shape):
        # Flatten input
        flat_input = input.view(-1)
        
        # Compute mean
        mean = flat_input.sum() / numel_to_reduce
        
        # Compute variance
        diff = flat_input - mean
        squared_diff = diff * diff
        sum_squared_diff = squared_diff.sum()
        var_val = sum_squared_diff / (numel_to_reduce - correction)
        
        # Compute standard deviation
        std_val = torch.sqrt(var_val)
        
        # Reshape result
        if keepdim:
            result = std_val.view(output_shape)
        else:
            result = std_val
        
        if out is not None:
            out.copy_(result)
            return out
        return result
    
    # For other cases, fall back to PyTorch implementation
    # This is a simplified fallback - in practice, you'd want a more sophisticated approach
    if out is not None:
        torch.std(input, dim=dim, correction=correction, keepdim=keepdim, out=out)
        return out
    return torch.std(input, dim=dim, correction=correction, keepdim=keepdim)