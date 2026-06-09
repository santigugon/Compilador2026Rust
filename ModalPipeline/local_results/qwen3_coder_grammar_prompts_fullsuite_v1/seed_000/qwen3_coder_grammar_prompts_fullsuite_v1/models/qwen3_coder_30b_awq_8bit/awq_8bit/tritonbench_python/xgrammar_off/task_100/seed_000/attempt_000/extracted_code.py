import torch
import triton
import triton.language as tl

@triton.jit
def _permute_copy_kernel(input_ptr, output_ptr, input_strides, output_strides, 
                        input_shape, output_shape, ndim: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    
    # Load input indices and compute output indices
    input_indices = tl.arange(0, BLOCK)
    output_indices = tl.zeros(BLOCK, dtype=tl.int64)
    
    # Compute output indices from input indices
    for i in range(ndim):
        # Get the dimension size for this axis
        dim_size = input_shape[i]
        # Get the stride for this axis in input
        input_stride = input_strides[i]
        # Get the stride for this axis in output
        output_stride = output_strides[i]
        
        # Compute the index in the flattened input
        input_idx = input_indices // input_stride
        input_indices = input_indices % input_stride
        
        # Compute the corresponding output index
        output_indices += input_idx * output_stride
    
    # Load from input and store to output
    mask = offsets < BLOCK
    input_val = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + output_indices, input_val, mask=mask)

def permute_copy(input, dims):
    # Validate dims
    if len(dims) != input.dim():
        raise ValueError("dims must have the same length as input tensor's dimensions")
    
    # Create output tensor with correct shape
    output_shape = tuple(input.shape[i] for i in dims)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Handle special case of no permutation
    if dims == tuple(range(input.dim())):
        return input.clone()
    
    # Compute strides for input and output
    input_strides = input.stride()
    output_strides = output.stride()
    
    # Flatten the input tensor for processing
    input_flat = input.contiguous().view(-1)
    output_flat = output.view(-1)
    
    # Launch kernel
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Create a simple kernel that copies elements with proper indexing
    @triton.jit
    def _permute_copy_simple_kernel(input_ptr, output_ptr, 
                                   input_shape, output_shape, 
                                   input_strides, output_strides,
                                   ndim: tl.constexpr, total_elements: tl.constexpr):
        pid = tl.program_id(0)
        offset = pid
        
        if offset >= total_elements:
            return
            
        # Convert linear index to multi-dimensional indices
        input_indices = tl.zeros(ndim, dtype=tl.int64)
        temp_offset = offset
        
        for i in range(ndim - 1, -1, -1):
            input_indices[i] = temp_offset % input_shape[i]
            temp_offset //= input_shape[i]
        
        # Convert multi-dimensional indices to output linear index
        output_offset = 0
        for i in range(ndim):
            output_offset += input_indices[i] * output_strides[i]
        
        # Copy element
        val = tl.load(input_ptr + offset)
        tl.store(output_ptr + output_offset, val)
    
    _permute_copy_simple_kernel[grid](
        input_flat, output_flat,
        input.shape, output.shape,
        input_strides, output_strides,
        ndim=input.dim(), total_elements=n
    )
    
    return output
