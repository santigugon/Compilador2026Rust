import torch
import triton
import triton.language as tl

@triton.jit
def fused_index_select_eq_kernel(
    input_ptr, 
    index_ptr, 
    other_ptr, 
    output_ptr,
    input_shape, 
    index_shape, 
    other_shape,
    dim,
    numel,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, numel)
    
    # Compute the output shape
    output_shape = []
    for i in range(len(input_shape)):
        if i == dim:
            output_shape.append(index_shape[0])
        else:
            output_shape.append(input_shape[i])
    
    # Compute the strides for input tensor
    input_strides = [1]
    for i in range(len(input_shape) - 1, 0, -1):
        input_strides = [input_strides[0] * input_shape[i]] + input_strides
    
    # Compute the strides for output tensor
    output_strides = [1]
    for i in range(len(output_shape) - 1, 0, -1):
        output_strides = [output_strides[0] * output_shape[i]] + output_strides
    
    # Compute the strides for index tensor
    index_strides = [1]
    for i in range(len(index_shape) - 1, 0, -1):
        index_strides = [index_strides[0] * index_shape[i]] + index_strides
    
    # Compute the strides for other tensor
    other_strides = [1]
    for i in range(len(other_shape) - 1, 0, -1):
        other_strides = [other_strides[0] * other_shape[i]] + other_strides
    
    for i in range(block_start, block_end):
        # Convert linear index to multi-dimensional index
        temp_idx = i
        output_indices = []
        for j in range(len(output_shape) - 1, -1, -1):
            if j == dim:
                output_indices = [temp_idx % output_shape[j]] + output_indices
                temp_idx //= output_shape[j]
            else:
                output_indices = [temp_idx % output_shape[j]] + output_indices
                temp_idx //= output_shape[j]
        
        # Compute the index in the input tensor
        input_indices = []
        for j in range(len(input_shape)):
            if j == dim:
                input_indices.append(output_indices[j])
            else:
                input_indices.append(output_indices[j])
        
        # Compute the index in the other tensor
        other_indices = []
        for j in range(len(other_shape)):
            if j == dim:
                other_indices.append(output_indices[j])
            else:
                other_indices.append(output_indices[j])
        
        # Compute the linear indices
        input_linear_idx = 0
        for j in range(len(input_indices)):
            input_linear_idx += input_indices[j] * input_strides[j]
        
        other_linear_idx = 0
        for j in range(len(other_indices)):
            other_linear_idx += other_indices[j] * other_strides[j]
        
        # Load values
        input_val = tl.load(input_ptr + input_linear_idx)
        other_val = tl.load(other_ptr + other_linear_idx)
        
        # Perform comparison
        result = input_val == other_val
        
        # Store result
        tl.store(output_ptr + i, result)

def fused_index_select_eq(input, dim, index, other, *, out=None):
    # Validate inputs
    if dim < 0:
        dim += input.dim()
    if dim >= input.dim():
        raise ValueError("dim out of range")
    
    # Handle scalar other
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure index is long tensor
    if index.dtype != torch.long:
        index = index.long()
    
    # Compute output shape
    output_shape = list(input.shape)
    output_shape[dim] = index.shape[0]
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=torch.bool, device=input.device)
    else:
        if out.shape != tuple(output_shape):
            raise ValueError("out tensor has incorrect shape")
    
    # Prepare input tensors for kernel
    input = input.contiguous()
    other = other.contiguous()
    
    # Compute total number of elements in output
    numel = 1
    for dim_size in output_shape:
        numel *= dim_size
    
    # Launch kernel
    if numel > 0:
        BLOCK_SIZE = 1024
        grid_size = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Prepare kernel arguments
        input_ptr = input.data_ptr()
        index_ptr = index.data_ptr()
        other_ptr = other.data_ptr()
        output_ptr = out.data_ptr()
        
        # Get shapes
        input_shape = list(input.shape)
        index_shape = list(index.shape)
        other_shape = list(other.shape)
        
        # Launch kernel
        fused_index_select_eq_kernel[grid_size](
            input_ptr, 
            index_ptr, 
            other_ptr, 
            output_ptr,
            input_shape, 
            index_shape, 
            other_shape,
            dim,
            numel,
            BLOCK_SIZE
        )
    
    return out
