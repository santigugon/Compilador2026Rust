import torch
import triton
import triton.language as tl

@triton.jit
def tile_exp_kernel(
    input_ptr, 
    output_ptr, 
    input_shape, 
    output_shape, 
    num_dims, 
    tile_dims, 
    BLOCK_SIZE
):
    # Compute global thread index
    global_id = tl.program_id(0)
    
    # Calculate total elements in output
    total_elements = 1
    for i in range(num_dims):
        total_elements *= output_shape[i]
    
    # Each thread processes one element
    if global_id >= total_elements:
        return
    
    # Convert linear index to multi-dimensional coordinates
    coords = [0] * num_dims
    temp_id = global_id
    for i in range(num_dims - 1, -1, -1):
        coords[i] = temp_id % output_shape[i]
        temp_id //= output_shape[i]
    
    # Calculate corresponding input coordinates
    input_coords = [0] * num_dims
    for i in range(num_dims):
        input_coords[i] = coords[i] % input_shape[i]
    
    # Compute linear indices
    input_idx = 0
    output_idx = 0
    stride_input = 1
    stride_output = 1
    
    for i in range(num_dims - 1, -1, -1):
        input_idx += input_coords[i] * stride_input
        output_idx += coords[i] * stride_output
        stride_input *= input_shape[i]
        stride_output *= output_shape[i]
    
    # Load input value, apply exp, and store result
    input_val = tl.load(input_ptr + input_idx)
    output_val = tl.exp(input_val)
    tl.store(output_ptr + output_idx, output_val)

def fused_tile_exp(input, dims, *, out=None):
    # Validate input
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if not isinstance(dims, tuple):
        raise TypeError("dims must be a tuple")
    
    # Prepare input tensor for Triton
    input = input.contiguous()
    
    # Calculate output shape
    input_shape = input.shape
    num_dims = len(input_shape)
    
    # Adjust dims to match input dimensions
    if len(dims) < num_dims:
        dims = (1,) * (num_dims - len(dims)) + dims
    elif len(dims) > num_dims:
        dims = dims[-num_dims:]
    
    # Calculate output shape
    output_shape = tuple(input_shape[i] * dims[i] for i in range(num_dims))
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        if out.shape != output_shape:
            raise ValueError("out tensor must have the same shape as the output")
    
    # Prepare for Triton kernel launch
    total_elements = 1
    for dim in output_shape:
        total_elements *= dim
    
    # Launch kernel
    if total_elements > 0:
        grid_size = (total_elements + 256 - 1) // 256
        tile_exp_kernel[grid_size](
            input_ptr=input.data_ptr(),
            output_ptr=out.data_ptr(),
            input_shape=input_shape,
            output_shape=output_shape,
            num_dims=num_dims,
            tile_dims=dims,
            BLOCK_SIZE=256
        )
    
    return out
