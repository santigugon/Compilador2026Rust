import torch
import triton
import triton.language as tl

@triton.jit
def _index_select_eq_kernel(
    input_ptr, index_ptr, other_ptr, out_ptr,
    input_shape0: tl.constexpr, input_shape1: tl.constexpr,
    index_size: tl.constexpr,
    dim: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    # Calculate the number of elements in the output
    output_elements = index_size
    if dim == 0:
        output_elements = index_size * input_shape1
    else:
        output_elements = input_shape0 * index_size
    
    # Each block processes BLOCK elements
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < output_elements
    
    # For simplicity, we'll process one element at a time
    # In a real implementation, we'd need to handle the indexing logic properly
    # This is a simplified version that assumes the indexing is handled by PyTorch
    # and we just do the comparison
    
    # Load the index values
    if dim == 0:
        # For dim=0, we're selecting rows
        indices = tl.load(index_ptr + offsets, mask=mask)
        # Load input elements
        input_offsets = indices * input_shape1 + tl.arange(0, 1)
        input_vals = tl.load(input_ptr + input_offsets, mask=mask)
        # Load other values
        other_vals = tl.load(other_ptr + offsets, mask=mask)
        # Perform comparison
        result = input_vals == other_vals
        tl.store(out_ptr + offsets, result, mask=mask)
    else:
        # For dim=1, we're selecting columns
        indices = tl.load(index_ptr + offsets, mask=mask)
        # Load input elements
        input_offsets = tl.arange(0, 1) * input_shape1 + indices
        input_vals = tl.load(input_ptr + input_offsets, mask=mask)
        # Load other values
        other_vals = tl.load(other_ptr + offsets, mask=mask)
        # Perform comparison
        result = input_vals == other_vals
        tl.store(out_ptr + offsets, result, mask=mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Handle scalar other
    if other.dim() == 0:
        other = other.expand_as(input)
    
    # Perform index selection using PyTorch
    selected = torch.index_select(input, dim, index)
    
    # Perform element-wise equality comparison
    if out is None:
        out = torch.empty_like(selected, dtype=torch.bool)
    
    # Use Triton kernel for the comparison
    n = selected.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Create a temporary tensor for the other values to match the selected shape
    if other.shape != selected.shape:
        # Broadcast other to match selected shape
        other_expanded = other.expand_as(selected)
    else:
        other_expanded = other
    
    # Allocate output tensor
    out = torch.empty_like(selected, dtype=torch.bool)
    
    # Launch kernel
    _index_select_eq_kernel[grid](
        selected, index, other_expanded, out,
        selected.shape[0], selected.shape[1],
        index.size(0),
        dim,
        BLOCK=block
    )
    
    return out
