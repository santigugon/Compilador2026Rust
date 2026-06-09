import torch
import triton
import triton.language as tl

def _get_strides(tensor):
    return [int(s) for s in tensor.stride()]

def _get_shape(tensor):
    return [int(s) for s in tensor.shape]

@triton.jit
def _fused_index_select_eq_kernel(
    input_ptr, index_ptr, other_ptr, out_ptr,
    input_strides, other_strides,
    index_size, input_shape, other_shape,
    dim, dim_size, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate output shape and strides
    output_shape = []
    output_strides = []
    
    # Build output shape by replacing dim with index_size
    for i, s in enumerate(input_shape):
        if i == dim:
            output_shape.append(index_size)
        else:
            output_shape.append(s)
    
    # Calculate output strides
    output_strides = [0] * len(output_shape)
    stride = 1
    for i in range(len(output_shape) - 1, -1, -1):
        output_strides[i] = stride
        stride *= output_shape[i]
    
    # Calculate global offset for this thread
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Calculate total output elements
    total_elements = 1
    for s in output_shape:
        total_elements *= s
    
    mask = offsets < total_elements
    
    # Convert linear offset to multi-dimensional indices
    indices = []
    temp_offset = offsets
    for i in range(len(output_shape) - 1, -1, -1):
        idx = temp_offset % output_shape[i]
        indices.append(idx)
        temp_offset //= output_shape[i]
    indices.reverse()
    
    # Get the index for the selected dimension
    selected_idx = indices[dim]
    
    # Load the index value
    index_val = tl.load(index_ptr + selected_idx, mask=selected_idx < index_size)
    
    # Calculate input position
    input_offset = 0
    for i in range(len(input_shape)):
        if i == dim:
            input_offset += index_val * input_strides[i]
        else:
            input_offset += indices[i] * input_strides[i]
    
    # Load input value
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Calculate other position
    other_offset = 0
    for i in range(len(other_shape)):
        if i < len(other_shape):
            other_offset += indices[i] * other_strides[i]
        else:
            other_offset += 0  # For broadcasting
    
    # Load other value
    other_val = tl.load(other_ptr + other_offset, mask=mask, other=0.0)
    
    # Perform equality comparison
    result = input_val == other_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def _fused_index_select_eq_kernel_scalar(
    input_ptr, index_ptr, other_val, out_ptr,
    input_strides, input_shape,
    index_size, dim, dim_size, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Calculate output shape and strides
    output_shape = []
    output_strides = []
    
    # Build output shape by replacing dim with index_size
    for i, s in enumerate(input_shape):
        if i == dim:
            output_shape.append(index_size)
        else:
            output_shape.append(s)
    
    # Calculate output strides
    output_strides = [0] * len(output_shape)
    stride = 1
    for i in range(len(output_shape) - 1, -1, -1):
        output_strides[i] = stride
        stride *= output_shape[i]
    
    # Calculate global offset for this thread
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Calculate total output elements
    total_elements = 1
    for s in output_shape:
        total_elements *= s
    
    mask = offsets < total_elements
    
    # Convert linear offset to multi-dimensional indices
    indices = []
    temp_offset = offsets
    for i in range(len(output_shape) - 1, -1, -1):
        idx = temp_offset % output_shape[i]
        indices.append(idx)
        temp_offset //= output_shape[i]
    indices.reverse()
    
    # Get the index for the selected dimension
    selected_idx = indices[dim]
    
    # Load the index value
    index_val = tl.load(index_ptr + selected_idx, mask=selected_idx < index_size)
    
    # Calculate input position
    input_offset = 0
    for i in range(len(input_shape)):
        if i == dim:
            input_offset += index_val * input_strides[i]
        else:
            input_offset += indices[i] * input_strides[i]
    
    # Load input value
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # Perform equality comparison with scalar
    result = input_val == other_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    if dim < 0:
        dim = input.dim() + dim
    
    # Handle scalar other case
    if not torch.is_tensor(other):
        scalar_other = other
        other = torch.tensor(scalar_other, dtype=input.dtype, device=input.device)
    
    # Get input shape and strides
    input_shape = _get_shape(input)
    input_strides = _get_strides(input)
    
    # Get other shape and strides
    other_shape = _get_shape(other)
    other_strides = _get_strides(other)
    
    # Get index size
    index_size = index.numel()
    
    # Calculate output shape
    output_shape = []
    for i, s in enumerate(input_shape):
        if i == dim:
            output_shape.append(index_size)
        else:
            output_shape.append(s)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=torch.bool, device=input.device)
    else:
        if out.shape != tuple(output_shape):
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {tuple(output_shape)}")
        if out.dtype != torch.bool:
            raise ValueError("Output tensor must have bool dtype")
    
    # Launch kernel
    block = 256
    total_elements = out.numel()
    grid = (triton.cdiv(total_elements, block),)
    
    if torch.is_tensor(other):
        _fused_index_select_eq_kernel[grid](
            input, index, other, out,
            input_strides, other_strides,
            index_size, input_shape, other_shape,
            dim, input_shape[dim], BLOCK=block
        )
    else:
        _fused_index_select_eq_kernel_scalar[grid](
            input, index, scalar_other, out,
            input_strides, input_shape,
            index_size, dim, input_shape[dim], BLOCK=block
        )
    
    return out