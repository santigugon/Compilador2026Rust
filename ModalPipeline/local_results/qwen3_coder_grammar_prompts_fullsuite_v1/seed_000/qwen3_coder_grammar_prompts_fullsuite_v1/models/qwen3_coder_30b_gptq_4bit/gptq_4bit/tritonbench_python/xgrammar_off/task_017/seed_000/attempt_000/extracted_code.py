import torch
import triton
import triton.language as tl

@triton.jit
def _index_select_eq_kernel(
    input_ptr, index_ptr, other_ptr, out_ptr,
    input_shape0: tl.constexpr, input_shape1: tl.constexpr,
    index_size: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Compute the total number of elements in the output
    total_elements = input_shape0 * index_size
    
    # Create mask for valid elements
    mask = offsets < total_elements
    
    # Calculate which index we're working with
    index_id = offsets // input_shape0
    element_id = offsets % input_shape0
    
    # Load index values
    index_val = tl.load(index_ptr + index_id, mask=index_id < index_size, other=0)
    
    # Load input values using index
    input_offsets = index_val * input_shape0 + element_id
    input_val = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Load other value (could be tensor or scalar)
    other_val = tl.load(other_ptr + element_id, mask=mask, other=0.0)
    
    # Perform equality comparison
    result = input_val == other_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    if dim < 0:
        dim = input.dim() + dim
    
    if dim >= input.dim():
        raise ValueError("dim must be within the range of input tensor dimensions")
    
    # Get the shape of input tensor
    input_shape = input.shape
    
    # Get the size of the indexing dimension
    index_size = index.size(0)
    
    # Create output tensor
    if out is None:
        # Create output tensor with appropriate shape
        output_shape = list(input_shape)
        output_shape[dim] = index_size
        out = torch.empty(output_shape, dtype=torch.bool, device=input.device)
    else:
        if out.shape != tuple(output_shape):
            raise ValueError("out tensor must have the same shape as the selected elements")
    
    # Handle scalar other case
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other tensor has the right shape for broadcasting
    if other.dim() == 0:
        other = other.expand(input_shape[0])
    elif other.shape[0] != input_shape[0]:
        raise ValueError("other tensor must have compatible shape for broadcasting")
    
    # Calculate total elements
    total_elements = input_shape[0] * index_size
    
    # Set block size
    BLOCK = 256
    grid = (triton.cdiv(total_elements, BLOCK),)
    
    # Launch kernel
    _index_select_eq_kernel[grid](
        input, index, other, out,
        input_shape[0], input_shape[1] if len(input_shape) > 1 else 1,
        index_size,
        BLOCK=BLOCK
    )
    
    return out
