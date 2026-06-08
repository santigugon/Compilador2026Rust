import torch
import triton
import triton.language as tl

@triton.jit
def _mean_row_kernel(x_ptr, out_ptr, n_rows: tl.constexpr, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n_rows:
        return
    
    # Each thread handles one row
    row_offset = pid * n_cols
    offsets = tl.arange(0, BLOCK)
    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    
    # Process the row in chunks
    for i in range(0, n_cols, BLOCK):
        chunk_offsets = row_offset + i + offsets
        mask = (i + offsets) < n_cols
        x_vals = tl.load(x_ptr + chunk_offsets, mask=mask, other=0.0)
        sum_val += x_vals
    
    # Compute mean for this row
    row_sum = tl.sum(sum_val, axis=0)
    mean_val = row_sum / n_cols
    
    # Store result
    tl.store(out_ptr + pid, mean_val)

def mean(input, dim, keepdim=False, dtype=None, out=None):
    # Handle dtype casting if needed
    if dtype is not None:
        input = input.to(dtype)
    
    # Handle scalar input case
    if input.dim() == 0:
        if out is not None:
            out.copy_(input)
        else:
            out = input.clone()
        return out
    
    # Handle dim as int
    if isinstance(dim, int):
        dim = [dim]
    
    # Normalize negative dimensions
    normalized_dims = []
    for d in dim:
        if d < 0:
            d = input.dim() + d
        normalized_dims.append(d)
    
    # Sort dimensions in descending order to avoid index shifting issues
    normalized_dims = sorted(normalized_dims, reverse=True)
    
    # Calculate output shape
    output_shape = list(input.shape)
    if keepdim:
        for d in normalized_dims:
            output_shape[d] = 1
    else:
        for d in normalized_dims:
            output_shape.pop(d)
    
    # Create output tensor
    if out is not None:
        if out.shape != torch.Size(output_shape):
            raise ValueError("out tensor has incorrect shape")
        out = out
    else:
        out = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    
    # Handle special case: no reduction needed
    if len(normalized_dims) == 0:
        if out is not None:
            out.copy_(input)
        else:
            out = input.clone()
        return out
    
    # Handle the case where we reduce over all dimensions
    if len(normalized_dims) == input.dim():
        # Compute total sum and divide by total elements
        total_elements = input.numel()
        if total_elements == 0:
            out.fill_(0)
        else:
            # Use a simple approach for all-dim reduction
            flat_input = input.flatten()
            sum_val = flat_input.sum()
            mean_val = sum_val / total_elements
            out.fill_(mean_val)
        return out
    
    # For row-wise mean computation, we need to handle the reduction properly
    # This is a simplified approach for the case where we reduce along one dimension
    if len(normalized_dims) == 1 and normalized_dims[0] == input.dim() - 1:
        # Reduce last dimension (row-wise mean)
        n_rows = 1
        n_cols = input.shape[-1]
        for i in range(input.dim() - 1):
            n_rows *= input.shape[i]
        
        if n_rows == 0:
            out.fill_(0)
            return out
            
        block = 256
        grid = (triton.cdiv(n_rows, 1),)
        
        # Create a flattened version for easier processing
        if n_cols > 0:
            _mean_row_kernel[grid](input, out, n_rows, n_cols, BLOCK=block)
        else:
            out.fill_(0)
        return out
    
    # For more complex cases, fall back to PyTorch
    # This is a simplified implementation that handles the most common case
    # For complex multi-dim reductions, we'll use PyTorch's implementation
    if out is not None:
        return torch.mean(input, dim=dim, keepdim=keepdim, out=out)
    else:
        return torch.mean(input, dim=dim, keepdim=keepdim)

##################################################################################################################################################



import torch

def test_mean():
    results = {}

    # Test case 1: Basic mean computation over a single dimension
    input_tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = mean(input_tensor1, dim=0)

    # Test case 2: Mean computation with keepdim=True
    input_tensor2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_2"] = mean(input_tensor2, dim=1, keepdim=True)

    # Test case 3: Mean computation over multiple dimensions
    input_tensor3 = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    results["test_case_3"] = mean(input_tensor3, dim=(0, 2))

    # Test case 4: Mean computation with dtype specified
    input_tensor4 = torch.tensor([[1, 2], [3, 4]], device='cuda', dtype=torch.int32)
    results["test_case_4"] = mean(input_tensor4, dim=0, dtype=torch.float32)

    return results

test_results = test_mean()
