import torch
import triton
import triton.language as tl

def _mean_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, n_rows: tl.constexpr, n_cols: tl.constexpr, keepdim: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_id = pid // n_cols
    col_id = pid % n_cols
    
    if row_id < n_rows and col_id < n_cols:
        # Load data for this row
        offsets = row_id * n_cols + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute mean for this row
        sum_val = tl.sum(x)
        mean_val = sum_val / n_cols
        
        # Store result
        if keepdim:
            # Store in the same shape as input
            out_offsets = row_id * n_cols + col_id
            tl.store(out_ptr + out_offsets, mean_val, mask=col_id < n_cols)
        else:
            # Store in reduced shape
            out_offsets = row_id
            tl.store(out_ptr + out_offsets, mean_val, mask=col_id == 0)

def _mean_kernel_multi_dim(x_ptr, out_ptr, n_elements: tl.constexpr, n_rows: tl.constexpr, n_cols: tl.constexpr, keepdim: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_id = pid // n_cols
    col_id = pid % n_cols
    
    if row_id < n_rows and col_id < n_cols:
        # Load data for this row
        offsets = row_id * n_cols + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute mean for this row
        sum_val = tl.sum(x)
        mean_val = sum_val / n_cols
        
        # Store result
        if keepdim:
            # Store in the same shape as input
            out_offsets = row_id * n_cols + col_id
            tl.store(out_ptr + out_offsets, mean_val, mask=col_id < n_cols)
        else:
            # Store in reduced shape
            out_offsets = row_id
            tl.store(out_ptr + out_offsets, mean_val, mask=col_id == 0)

def mean(input, dim, keepdim=False, dtype=None, out=None):
    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
    
    # Handle scalar input
    if input.dim() == 0:
        if out is not None:
            out.copy_(input)
        return input
    
    # Handle single dimension
    if isinstance(dim, int):
        # Get dimensions
        input_size = input.size()
        n_rows = 1
        n_cols = 1
        
        # Calculate total elements
        n_elements = input.numel()
        
        # Calculate rows and columns
        if dim < 0:
            dim = len(input_size) + dim
        
        # Compute output shape
        output_shape = list(input_size)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)
        
        # Create output tensor
        if out is not None:
            out = out.resize_(output_shape)
        else:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        
        # Handle case where we're reducing over the last dimension
        if dim == len(input_size) - 1:
            n_rows = input_size[0] if len(input_size) > 1 else 1
            n_cols = input_size[-1]
        else:
            # For other dimensions, we need to compute the reduction differently
            # This is a simplified approach for the general case
            n_rows = 1
            n_cols = input_size[dim]
        
        # Launch kernel
        block = 256
        grid = (triton.cdiv(n_elements, block),)
        
        # For simplicity, we'll use a basic approach
        if out is not None:
            out = out.resize_(output_shape)
        else:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        
        # Use PyTorch for complex cases
        if dim != len(input_size) - 1:
            # Fall back to PyTorch for complex cases
            if keepdim:
                return input.mean(dim=dim, keepdim=keepdim)
            else:
                return input.mean(dim=dim, keepdim=keepdim)
        
        # Simple case: reduce last dimension
        _mean_kernel[grid](input, out, n_elements, n_rows, n_cols, keepdim, BLOCK=block)
        return out
    
    # Handle multiple dimensions
    elif isinstance(dim, (tuple, list)):
        # For multiple dimensions, fall back to PyTorch
        if out is not None:
            return input.mean(dim=dim, keepdim=keepdim, out=out)
        else:
            return input.mean(dim=dim, keepdim=keepdim)
    
    # Default case
    else:
        if out is not None:
            return input.mean(keepdim=keepdim, out=out)
        else:
            return input.mean(keepdim=keepdim)