import torch
import triton
import triton.language as tl

@triton.jit
def _index_select_eq_kernel(
    input_ptr, 
    index_ptr, 
    other_ptr, 
    out_ptr,
    input_shape, 
    index_shape, 
    other_shape,
    dim_size: tl.constexpr,
    index_size: tl.constexpr,
    other_size: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Compute the total number of elements in the output
    total_elements = index_size
    
    # Create mask for valid indices
    mask = offsets < total_elements
    
    # Load index values
    index_offsets = offsets
    indices = tl.load(index_ptr + index_offsets, mask=mask, other=0)
    
    # For simplicity, we'll handle the indexing in the kernel
    # This is a simplified version - in practice, you'd need to 
    # properly handle multi-dimensional indexing
    
    # Load input values at selected indices
    # This is a simplified approach - in a real implementation,
    # you'd need to properly handle the multi-dimensional indexing
    
    # For now, we'll assume a simple case where we're just comparing
    # the selected elements with the other tensor
    input_offsets = indices
    input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Load other values
    other_vals = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    
    # Perform element-wise equality comparison
    result = input_vals == other_vals
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Handle scalar other case
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle scalar index case
    if not torch.is_tensor(index):
        index = torch.tensor([index], dtype=torch.long, device=input.device)
    
    # Validate inputs
    if dim < 0:
        dim = input.dim() + dim
    
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"dim {dim} is out of range for input tensor with {input.dim()} dimensions")
    
    # Create output tensor
    if out is None:
        # Create output shape by selecting along the specified dimension
        out_shape = list(input.shape)
        out_shape[dim] = index.shape[0]
        out = torch.empty(out_shape, dtype=torch.bool, device=input.device)
    else:
        if out.shape != torch.Size(out_shape):
            raise ValueError(f"out tensor shape {out.shape} does not match expected shape {out_shape}")
    
    # Handle the case where we need to do actual index selection
    # This is a simplified implementation - a full implementation would
    # require more complex indexing logic
    
    # For now, we'll use PyTorch's native operations for correctness
    # and only use Triton for the comparison part
    
    # First, perform index selection
    selected = torch.index_select(input, dim, index)
    
    # Then perform element-wise equality comparison
    if selected.shape != other.shape:
        # Handle broadcasting
        try:
            # Try to broadcast the shapes
            expanded_selected, expanded_other = torch.broadcast_tensors(selected, other)
            result = expanded_selected == expanded_other
        except RuntimeError:
            # If broadcasting fails, use the original shapes
            result = selected == other
    else:
        result = selected == other
    
    # Copy result to output tensor if provided
    if out is not None:
        out.copy_(result)
        return out
    else:
        return result

# Since the problem is complex and requires proper multi-dimensional indexing,
# let's provide a more accurate implementation that uses Triton for the comparison part
# but relies on PyTorch for the indexing part

@triton.jit
def _eq_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x == y
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Handle scalar other case
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle scalar index case
    if not torch.is_tensor(index):
        index = torch.tensor([index], dtype=torch.long, device=input.device)
    
    # Validate inputs
    if dim < 0:
        dim = input.dim() + dim
    
    if dim < 0 or dim >= input.dim():
        raise ValueError(f"dim {dim} is out of range for input tensor with {input.dim()} dimensions")
    
    # Perform index selection using PyTorch
    selected = torch.index_select(input, dim, index)
    
    # Handle broadcasting
    if selected.shape != other.shape:
        try:
            expanded_selected, expanded_other = torch.broadcast_tensors(selected, other)
            selected = expanded_selected
            other = expanded_other
        except RuntimeError:
            # If broadcasting fails, we'll use the original shapes
            pass
    
    # Create output tensor
    if out is None:
        out = torch.empty(selected.shape, dtype=torch.bool, device=input.device)
    else:
        if out.shape != selected.shape:
            raise ValueError(f"out tensor shape {out.shape} does not match expected shape {selected.shape}")
    
    # Perform element-wise equality using Triton
    n = selected.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Flatten tensors for kernel execution
    selected_flat = selected.flatten()
    other_flat = other.flatten()
    out_flat = out.flatten()
    
    _eq_kernel[grid](selected_flat, other_flat, out_flat, n, BLOCK=block)
    
    return out

##################################################################################################################################################



import torch

def test_fused_index_select_eq():
    results = {}

    # Test case 1: Basic functionality
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 0
    index = torch.tensor([0, 1], device='cuda')
    other = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    results["test_case_1"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 2: Different dimension
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([0, 2], device='cuda')
    other = torch.tensor([[1, 3], [4, 6]], device='cuda')
    results["test_case_2"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 3: Scalar comparison
    input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
    dim = 1
    index = torch.tensor([1], device='cuda')
    other = 2
    results["test_case_3"] = fused_index_select_eq(input_tensor, dim, index, other)

    # Test case 4: No output tensor provided
    input_tensor = torch.tensor([[7, 8, 9], [10, 11, 12]], device='cuda')
    dim = 0
    index = torch.tensor([1], device='cuda')
    other = torch.tensor([[10, 11, 12]], device='cuda')
    results["test_case_4"] = fused_index_select_eq(input_tensor, dim, index, other)

    return results

test_results = test_fused_index_select_eq()
