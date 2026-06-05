import torch
import triton
import triton.language as tl

@triton.jit
def _mean_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, n_rows: tl.constexpr, row_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row_id = pid // (n_elements // row_size)
    col_id = pid % (n_elements // row_size)
    
    # Each block handles one row
    if row_id >= n_rows:
        return
    
    # Compute mean for this row
    sum_val = 0.0
    for i in range(0, row_size, BLOCK):
        offsets = row_id * row_size + i + tl.arange(0, BLOCK)
        mask = offsets < (row_id + 1) * row_size
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x)
    
    mean_val = sum_val / row_size
    tl.store(out_ptr + row_id, mean_val)

def mean(input, dim, keepdim=False, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)
    
    # Handle scalar input
    if input.dim() == 0:
        if out is not None:
            out.copy_(input)
        return input
    
    # Handle list/tuple of dimensions
    if isinstance(dim, (list, tuple)):
        # For multiple dimensions, we need to flatten and then reduce
        # This is a simplified approach - in practice, you'd want to handle this more carefully
        # For now, we'll reduce one dimension at a time
        input = input.clone()
        for d in sorted(dim, reverse=True):
            input = input.mean(dim=d, keepdim=True)
        if not keepdim:
            input = input.squeeze(dim)
        if out is not None:
            out.copy_(input)
        return input
    
    # Single dimension case
    if dim < 0:
        dim = input.dim() + dim
    
    # Special case: reduce all dimensions
    if dim is None or dim == -1:
        if out is not None:
            out.copy_(input.mean())
        return input.mean()
    
    # Handle the case where we reduce over the last dimension
    if dim == input.dim() - 1:
        # Create output tensor
        if keepdim:
            output_shape = list(input.shape)
            output_shape[dim] = 1
        else:
            output_shape = [s for i, s in enumerate(input.shape) if i != dim]
        
        out_tensor = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        
        # Launch kernel
        n_elements = input.numel()
        n_rows = input.shape[0] if input.dim() > 1 else 1
        row_size = input.shape[-1] if input.dim() > 0 else 1
        
        block = 256
        grid = triton.cdiv(n_elements, block)
        
        _mean_kernel[grid](input, out_tensor, n_elements, n_rows, row_size, BLOCK=block)
        
        if out is not None:
            out.copy_(out_tensor)
        return out_tensor
    
    # For other dimensions, we need to handle the reduction more carefully
    # This is a simplified version that works for the common case
    if keepdim:
        output_shape = list(input.shape)
        output_shape[dim] = 1
    else:
        output_shape = [s for i, s in enumerate(input.shape) if i != dim]
    
    out_tensor = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # For simplicity, we'll use PyTorch's implementation for complex cases
    # and only use Triton for the simple case of reducing last dimension
    if dim == input.dim() - 1:
        # Use our Triton kernel for last dimension reduction
        n_elements = input.numel()
        n_rows = input.shape[0] if input.dim() > 1 else 1
        row_size = input.shape[-1] if input.dim() > 0 else 1
        
        block = 256
        grid = triton.cdiv(n_elements, block)
        
        _mean_kernel[grid](input, out_tensor, n_elements, n_rows, row_size, BLOCK=block)
    else:
        # Fall back to PyTorch for other dimensions
        result = input.mean(dim=dim, keepdim=keepdim)
        if out is not None:
            out.copy_(result)
        return result
    
    if out is not None:
        out.copy_(out_tensor)
    return out_tensor
