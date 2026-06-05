import torch
import triton
import triton.language as tl

@triton.jit
def min_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    min_val = tl.minimum(input, tl.load(output_ptr + offsets, mask=mask))
    min_idx = tl.where(input < tl.load(output_ptr + offsets, mask=mask), 
                       offsets, tl.load(indices_ptr + offsets, mask=mask))
    tl.store(output_ptr + offsets, min_val, mask=mask)
    tl.store(indices_ptr + offsets, min_idx, mask=mask)

@triton.jit
def min_row_kernel(
    input_ptr,
    output_ptr,
    indices_ptr,
    rows,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row = pid // cols
    col = pid % cols
    if row < rows:
        input_row = input_ptr + row * cols
        output_row = output_ptr + row
        indices_row = indices_ptr + row
        min_val = tl.load(input_row + col)
        min_idx = col
        for i in range(1, cols):
            val = tl.load(input_row + i)
            if val < min_val:
                min_val = val
                min_idx = i
        tl.store(output_row, min_val)
        tl.store(indices_row, min_idx)

def min(input, dim, keepdim=False, *, out=None):
    if out is not None:
        raise NotImplementedError("out parameter is not supported")
    
    if dim < 0:
        dim = input.dim() + dim
    
    if dim >= input.dim():
        raise ValueError("dim out of range")
    
    if input.dim() == 1:
        # For 1D tensor, just return the min and its index
        min_val = input.min()
        min_idx = input.argmin()
        if keepdim:
            min_val = min_val.unsqueeze(dim)
            min_idx = min_idx.unsqueeze(dim)
        return min_val, min_idx
    
    # For multi-dimensional tensors, use Triton kernel
    shape = input.shape
    if keepdim:
        output_shape = list(shape)
        output_shape[dim] = 1
    else:
        output_shape = [shape[i] for i in range(len(shape)) if i != dim]
    
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
    
    # Flatten the tensor for processing
    if dim == 0:
        # If reducing along first dimension, we need to process each row
        rows = shape[0]
        cols = shape[1] if len(shape) > 1 else 1
        if cols == 1:
            # Special case: single column
            output = input.min(dim=0, keepdim=keepdim)
            return output[0], output[1]
        else:
            # General case: multiple columns
            grid = (rows * cols + 255) // 256
            min_row_kernel[grid](input, output, indices, rows, cols, BLOCK_SIZE=256)
    else:
        # For other dimensions, we can use a simpler approach
        # This is a simplified implementation for demonstration
        if keepdim:
            output = input.min(dim=dim, keepdim=True)
            indices = input.argmin(dim=dim, keepdim=True)
        else:
            output = input.min(dim=dim, keepdim=False)
            indices = input.argmin(dim=dim, keepdim=False)
    
    return output, indices
