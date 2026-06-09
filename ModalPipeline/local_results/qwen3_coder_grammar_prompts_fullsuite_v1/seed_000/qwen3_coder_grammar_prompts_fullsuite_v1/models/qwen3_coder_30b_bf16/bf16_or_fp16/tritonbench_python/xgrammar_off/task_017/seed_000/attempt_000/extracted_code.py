import torch
import triton
import triton.language as tl

@triton.jit
def fused_index_select_eq_kernel(
    input_ptr, index_ptr, other_ptr, output_ptr,
    input_shape, index_shape, output_shape,
    dim, num_indices,
    BLOCK_SIZE: tl.constexpr
):
    # Compute global thread index
    pid = tl.program_id(0)
    num_elements = tl.numel(output_shape)
    
    # Each thread processes one element of the output
    if pid >= num_elements:
        return
    
    # Convert linear index to multi-dimensional index
    output_idx = pid
    input_idx = [0] * len(input_shape)
    
    # Compute the multi-dimensional index in output tensor
    temp = output_idx
    for i in range(len(output_shape) - 1, -1, -1):
        input_idx[i] = temp % output_shape[i]
        temp //= output_shape[i]
    
    # Compute the corresponding index in the input tensor
    # For the indexed dimension, use the index tensor
    input_idx[dim] = tl.load(index_ptr + input_idx[dim])
    
    # Compute linear index in input tensor
    input_linear_idx = 0
    stride = 1
    for i in range(len(input_shape) - 1, -1, -1):
        input_linear_idx += input_idx[i] * stride
        stride *= input_shape[i]
    
    # Load input element
    input_val = tl.load(input_ptr + input_linear_idx)
    
    # Load other element (scalar or tensor)
    other_val = tl.load(other_ptr + input_idx[dim]) if other_ptr.dtype == input_ptr.dtype else other_ptr
    
    # Perform equality comparison
    result = input_val == other_val
    
    # Store result
    tl.store(output_ptr + pid, result)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    if dim < 0:
        dim += input.dim()
    
    if dim < 0 or dim >= input.dim():
        raise ValueError("dim out of range")
    
    # Handle scalar other
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Compute output shape
    output_shape = list(input.shape)
    output_shape[dim] = index.shape[0]
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=torch.bool, device=input.device)
    else:
        if out.shape != tuple(output_shape):
            raise ValueError("out tensor has incorrect shape")
        if out.dtype != torch.bool:
            raise ValueError("out tensor must be boolean")
    
    # Prepare input tensors for kernel
    input_ptr = input.data_ptr()
    index_ptr = index.data_ptr()
    other_ptr = other.data_ptr()
    
    # Compute total elements in output
    num_elements = 1
    for s in output_shape:
        num_elements *= s
    
    # Launch kernel
    grid = (num_elements + 1024 - 1) // 1024
    fused_index_select_eq_kernel[grid](
        input_ptr, index_ptr, other_ptr, out.data_ptr(),
        input.shape, index.shape, output_shape,
        dim, index.shape[0],
        BLOCK_SIZE=1024
    )
    
    return out
