import torch
import triton
import triton.language as tl

@triton.jit
def _min_row_kernel(input_ptr, output_ptr, indices_ptr, n_rows: tl.constexpr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    # Each program handles one row
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    
    # Load the row
    offsets = row_id * n_cols + tl.arange(0, BLOCK)
    mask = offsets < n_rows * n_cols
    
    # Initialize min and index
    min_val = tl.full([1], float('inf'), dtype=tl.float32)
    min_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Process the row in chunks
    for i in range(0, n_cols, BLOCK):
        # Compute offsets for this chunk
        chunk_offsets = offsets + i
        chunk_mask = chunk_offsets < n_rows * n_cols
        
        # Load values
        values = tl.load(input_ptr + chunk_offsets, mask=chunk_mask, other=0.0)
        
        # Find min and index
        for j in range(BLOCK):
            if chunk_mask[j]:
                val = values[j]
                if val < min_val:
                    min_val = val
                    min_idx = chunk_offsets[j] - row_id * n_cols
    
    # Store results
    tl.store(output_ptr + row_id, min_val)
    tl.store(indices_ptr + row_id, min_idx)

def min(input, dim, keepdim=False, *, out=None):
    # Handle negative dimension
    if dim < 0:
        dim = input.dim() + dim
    
    # Get dimensions
    shape = input.shape
    n_rows = 1
    n_cols = 1
    
    # Calculate number of rows and columns
    for i in range(len(shape)):
        if i == dim:
            n_cols = shape[i]
        else:
            n_rows *= shape[i]
    
    # Create output tensors
    if out is not None:
        min_out, indices_out = out
    else:
        if keepdim:
            output_shape = list(shape)
            output_shape[dim] = 1
        else:
            output_shape = [shape[i] for i in range(len(shape)) if i != dim]
        
        min_out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        indices_out = torch.empty(output_shape, dtype=torch.long, device=input.device)
    
    # Handle special case where there's only one element
    if n_rows == 1 and n_cols == 1:
        min_out = input.clone()
        indices_out = torch.zeros_like(indices_out)
        if out is not None:
            out[0].copy_(min_out)
            out[1].copy_(indices_out)
        return (min_out, indices_out)
    
    # For small tensors, use PyTorch's implementation
    if n_rows * n_cols <= 1024:
        if out is not None:
            torch.min(input, dim=dim, keepdim=keepdim, out=out)
            return out
        else:
            return torch.min(input, dim=dim, keepdim=keepdim)
    
    # Use Triton kernel for larger tensors
    block = 256
    grid = (n_rows,)
    
    # Ensure input is contiguous
    input_contiguous = input.contiguous()
    
    # Allocate output tensors
    if out is None:
        if keepdim:
            output_shape = list(shape)
            output_shape[dim] = 1
        else:
            output_shape = [shape[i] for i in range(len(shape)) if i != dim]
        
        min_out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        indices_out = torch.empty(output_shape, dtype=torch.long, device=input.device)
    
    # Launch kernel
    _min_row_kernel[grid](input_contiguous, min_out, indices_out, n_rows, n_cols, BLOCK=block)
    
    if out is not None:
        out[0].copy_(min_out)
        out[1].copy_(indices_out)
        return out
    else:
        return (min_out, indices_out)
