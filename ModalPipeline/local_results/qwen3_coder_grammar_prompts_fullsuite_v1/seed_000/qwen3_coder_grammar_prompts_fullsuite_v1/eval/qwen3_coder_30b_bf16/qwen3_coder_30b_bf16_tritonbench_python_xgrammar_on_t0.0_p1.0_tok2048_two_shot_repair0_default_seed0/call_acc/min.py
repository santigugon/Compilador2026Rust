import torch
import triton
import triton.language as tl

def _min_kernel(input_ptr, output_ptr, indices_ptr, n_rows, n_cols, dim, keepdim, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if dim == 0:
        # Reduce along rows
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        
        # Load data for this row
        row_data = tl.load(input_ptr + pid * n_cols + offsets, mask=mask, other=tl.float32(1e38))
        
        # Find minimum and its index
        min_val = tl.min(row_data)
        min_idx = tl.argmin(row_data)
        
        # Store results
        tl.store(output_ptr + pid, min_val)
        tl.store(indices_ptr + pid, min_idx)
    else:
        # Reduce along columns
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_rows
        
        # Load data for this column
        col_data = tl.load(input_ptr + offsets * n_cols + pid, mask=mask, other=tl.float32(1e38))
        
        # Find minimum and its index
        min_val = tl.min(col_data)
        min_idx = tl.argmin(col_data)
        
        # Store results
        tl.store(output_ptr + pid, min_val)
        tl.store(indices_ptr + pid, min_idx)

def min(input, dim, keepdim=False, *, out=None):
    if out is not None:
        output, indices = out
    else:
        if keepdim:
            output_shape = list(input.shape)
            output_shape[dim] = 1
            output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
            indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
        else:
            output_shape = list(input.shape)
            output_shape.pop(dim)
            output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
            indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
    
    if input.numel() == 0:
        return output, indices
    
    # Handle the case where we reduce along the first dimension
    if dim == 0:
        n_rows = input.shape[0]
        n_cols = input.shape[1]
        block_size = 256
        grid_size = triton.cdiv(n_rows, block_size)
        _min_kernel[grid_size](input, output, indices, n_rows, n_cols, dim, keepdim, BLOCK_SIZE=block_size)
    else:
        n_rows = input.shape[0]
        n_cols = input.shape[1]
        block_size = 256
        grid_size = triton.cdiv(n_cols, block_size)
        _min_kernel[grid_size](input, output, indices, n_rows, n_cols, dim, keepdim, BLOCK_SIZE=block_size)
    
    if out is not None:
        return out
    else:
        return output, indices
##################################################################################################################################################



import torch

def test_min():
    results = {}

    # Test case 1: 2D tensor, dim=0, keepdim=False
    input_tensor = torch.tensor([[1, 2, 3], [4, 0, 6]], device='cuda')
    results["test_case_1"] = min(input_tensor, dim=0)

    # Test case 2: 2D tensor, dim=1, keepdim=False
    input_tensor = torch.tensor([[1, 2, 3], [4, 0, 6]], device='cuda')
    results["test_case_2"] = min(input_tensor, dim=1)

    # Test case 3: 3D tensor, dim=2, keepdim=True
    input_tensor = torch.tensor([[[1, 2, 3], [4, 0, 6]], [[7, 8, 9], [10, 11, 12]]], device='cuda')
    results["test_case_3"] = min(input_tensor, dim=2, keepdim=True)

    # Test case 4: 1D tensor, dim=0, keepdim=False
    input_tensor = torch.tensor([1, 2, 3, 0, 4, 5], device='cuda')
    results["test_case_4"] = min(input_tensor, dim=0)

    return results

test_results = test_min()
