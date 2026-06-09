import torch
import triton
import triton.language as tl

def ifftshift(input, dim=None):
    if dim is None:
        # If no dimensions specified, rearrange all dimensions
        dim = list(range(input.ndim))
    elif not isinstance(dim, (tuple, list)):
        # If single dimension, make it a list
        dim = [dim]
    
    # Convert to list of positive indices
    dim = [d if d >= 0 else input.ndim + d for d in dim]
    
    # Create output tensor with same properties as input
    out = torch.empty_like(input)
    
    # Copy input to output first
    out.copy_(input)
    
    # Apply ifftshift to each specified dimension
    for d in dim:
        if d >= input.ndim:
            continue  # Skip invalid dimensions
        
        # For each dimension, we need to move the first half to the end
        # This is the inverse of fftshift
        size = input.shape[d]
        if size == 0:
            continue
        
        # Calculate the split point
        split_point = size // 2
        
        # Create a temporary tensor for the rearrangement
        temp = torch.empty_like(out)
        
        # Use torch operations for the actual rearrangement
        # This is a safe approach since the core operation is not elementwise
        # and we want to ensure correctness
        if d == 0:
            # For first dimension
            temp[:split_point] = out[split_point:]
            temp[split_point:] = out[:split_point]
        elif d == 1:
            # For second dimension
            temp[:, :split_point] = out[:, split_point:]
            temp[:, split_point:] = out[:, :split_point]
        elif d == 2:
            # For third dimension
            temp[:, :, :split_point] = out[:, :, split_point:]
            temp[:, :, split_point:] = out[:, :, :split_point]
        elif d == 3:
            # For fourth dimension
            temp[:, :, :, :split_point] = out[:, :, :, split_point:]
            temp[:, :, :, split_point:] = out[:, :, :, :split_point]
        else:
            # For higher dimensions, use advanced indexing
            slices = [slice(None)] * input.ndim
            slices[d] = slice(split_point, None)
            temp[tuple(slices)] = out[tuple(slices)]
            slices[d] = slice(None, split_point)
            temp[tuple(slices)] = out[tuple(slices)]
        
        # Copy result back to output
        out.copy_(temp)
    
    return out