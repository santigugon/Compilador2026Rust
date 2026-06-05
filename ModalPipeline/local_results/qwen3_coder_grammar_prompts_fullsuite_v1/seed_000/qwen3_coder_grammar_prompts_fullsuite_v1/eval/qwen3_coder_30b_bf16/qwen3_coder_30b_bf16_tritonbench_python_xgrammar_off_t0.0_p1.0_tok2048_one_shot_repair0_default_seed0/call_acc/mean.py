import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    input_ptr, 
    output_ptr, 
    input_row_stride, 
    output_row_stride, 
    n_cols, 
    n_rows, 
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Load data and compute sum
    for col_start in range(0, n_cols, BLOCK_SIZE):
        col_offset = col_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offset < n_cols
        
        # Load input data
        input_data = tl.load(input_ptr + row_idx * input_row_stride + col_offset, mask=mask, other=0.0)
        
        # Accumulate sum
        acc += tl.sum(input_data)
    
    # Compute mean
    mean_val = acc / n_cols
    
    # Store result
    tl.store(output_ptr + row_idx * output_row_stride, mean_val)

def mean_triton(input, dim, keepdim=False, dtype=None, out=None):
    if isinstance(dim, int):
        dim = (dim,)
    
    # Handle negative dimensions
    dim = tuple(d if d >= 0 else input.dim() + d for d in dim)
    
    # Validate dimensions
    if not all(0 <= d < input.dim() for d in dim):
        raise ValueError("Dimension out of range")
    
    # Handle dtype casting
    if dtype is not None:
        input = input.to(dtype)
    
    # Create output shape
    output_shape = list(input.shape)
    if keepdim:
        for d in dim:
            output_shape[d] = 1
    else:
        for d in sorted(dim, reverse=True):
            output_shape.pop(d)
    
    # Create output tensor
    if out is not None:
        output = out
        if output.shape != tuple(output_shape):
            raise ValueError("Output tensor shape mismatch")
    else:
        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    
    # Handle reduction over multiple dimensions
    if len(dim) == 0:
        return input.clone()
    
    # For simplicity, we'll handle single dimension reduction
    if len(dim) == 1:
        reduce_dim = dim[0]
        if reduce_dim < 0:
            reduce_dim += input.dim()
        
        # Get dimensions
        n_rows = 1
        n_cols = input.shape[reduce_dim]
        for i in range(reduce_dim):
            n_rows *= input.shape[i]
        for i in range(reduce_dim + 1, input.dim()):
            n_cols *= input.shape[i]
        
        # Create output tensor if needed
        if out is None:
            output = torch.empty(input.shape[:reduce_dim] + input.shape[reduce_dim+1:], 
                               device=input.device, dtype=input.dtype)
        
        # Launch kernel
        BLOCK_SIZE = 1024
        grid = (n_rows,)
        
        mean_kernel[grid](
            input_ptr=input.data_ptr(),
            output_ptr=output.data_ptr(),
            input_row_stride=input.stride(0) if input.dim() > 1 else 1,
            output_row_stride=output.stride(0) if output.dim() > 0 else 1,
            n_cols=n_cols,
            n_rows=n_rows,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # For multiple dimensions, we'll reduce one by one
        # This is a simplified approach for demonstration
        temp = input
        sorted_dims = sorted(dim, reverse=True)
        for d in sorted_dims:
            temp = mean_triton(temp, d, keepdim=True)
        
        if not keepdim:
            # Remove the reduced dimensions
            output_shape = list(input.shape)
            for d in sorted(dim, reverse=True):
                output_shape.pop(d)
            output = temp.view(output_shape)
        else:
            output = temp
    
    return output

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
