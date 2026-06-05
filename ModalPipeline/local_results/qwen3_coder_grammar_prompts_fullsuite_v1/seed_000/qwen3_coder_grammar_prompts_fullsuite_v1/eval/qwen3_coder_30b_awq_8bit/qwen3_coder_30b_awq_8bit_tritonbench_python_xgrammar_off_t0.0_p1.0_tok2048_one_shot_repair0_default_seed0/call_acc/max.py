import torch
import triton
import triton.language as tl

@triton.jit
def max_kernel(
    input_ptr, output_ptr, indices_ptr, 
    input_row_stride, output_row_stride, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    input_row = input_ptr + row_idx * input_row_stride
    output_row = output_ptr + row_idx * output_row_stride
    indices_row = indices_ptr + row_idx * output_row_stride
    
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)
    
    for col in range(0, n_cols, BLOCK_SIZE):
        offsets = col + col_offsets
        mask = offsets < n_cols
        vals = tl.load(input_row + offsets, mask=mask, other=-float('inf'))
        
        new_max = tl.maximum(max_val, vals)
        new_idx = tl.where(vals > max_val, offsets, max_idx)
        
        max_val = new_max
        max_idx = new_idx
    
    # Reduce within block
    final_max = tl.max(max_val, axis=0)
    final_idx = tl.load(indices_ptr + row_idx * output_row_stride + tl.argmax(max_val, axis=0))
    
    tl.store(output_row, final_max)
    tl.store(indices_row, final_idx)

def max(input, dim, keepdim=False, *, out=None):
    if dim < 0:
        dim = input.dim() + dim
    
    if dim >= input.dim():
        raise ValueError("dim out of range")
    
    input = input.contiguous()
    
    if keepdim:
        output_shape = list(input.shape)
        output_shape[dim] = 1
    else:
        output_shape = list(input.shape)
        output_shape.pop(dim)
    
    output = torch.empty(output_shape, dtype=torch.float32, device=input.device)
    indices = torch.empty(output_shape, dtype=torch.long, device=input.device)
    
    if out is not None:
        output = out[0]
        indices = out[1]
    
    if input.numel() == 0:
        return (output, indices)
    
    BLOCK_SIZE = 1024
    n_rows = input.shape[0] if dim == 0 else input.numel() // input.shape[dim]
    n_cols = input.shape[dim]
    
    grid = (n_rows,)
    
    max_kernel[grid](
        input, output, indices,
        input.stride(0) if dim == 0 else input.stride(1),
        output.stride(0) if keepdim else 1,
        n_cols,
        BLOCK_SIZE
    )
    
    return (output, indices)

##################################################################################################################################################



import torch

def test_max():
    results = {}

    # Test case 1: Basic test with a 2D tensor
    input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
    results['test_case_1'] = max(input_tensor, dim=0)

    # Test case 2: Test with keepdim=True
    input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
    results['test_case_2'] = max(input_tensor, dim=1, keepdim=True)

    # Test case 3: Test with a 3D tensor
    input_tensor = torch.tensor([[[1, 3, 2], [4, 6, 5]], [[7, 9, 8], [10, 12, 11]]], device='cuda')
    results['test_case_3'] = max(input_tensor, dim=2)

    # Test case 4: Test with a negative dimension
    input_tensor = torch.tensor([[1, 3, 2], [4, 6, 5]], device='cuda')
    results['test_case_4'] = max(input_tensor, dim=-1)

    return results

test_results = test_max()
