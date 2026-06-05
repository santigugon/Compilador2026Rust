import torch
import triton
import triton.language as tl

@triton.jit
def broadcast_kernel(
    input_ptr, output_ptr, 
    size, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    input_data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_data, mask=mask)

def broadcast_tensors(*tensors):
    # Determine the maximum shape after broadcasting
    shapes = [tensor.shape for tensor in tensors]
    max_shape = []
    
    # Find the maximum dimension for each axis
    max_dims = []
    for i in range(max(len(shape) for shape in shapes)):
        max_dim = 0
        for shape in shapes:
            if i < len(shape):
                max_dim = max(max_dim, shape[i])
        max_dims.append(max_dim)
    
    # Create broadcasted tensors
    broadcasted_tensors = []
    for tensor in tensors:
        # Create output tensor with broadcasted shape
        output_shape = []
        for i, dim in enumerate(max_dims):
            if i < len(tensor.shape):
                if tensor.shape[i] == 1:
                    output_shape.append(dim)
                else:
                    output_shape.append(tensor.shape[i])
            else:
                output_shape.append(dim)
        
        # Allocate output tensor
        output_tensor = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        
        # Use Triton kernel for broadcasting
        if tensor.numel() > 0:
            # Calculate grid size
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(tensor.numel(), BLOCK_SIZE),)
            
            # Launch kernel
            broadcast_kernel[grid](
                tensor.data_ptr(),
                output_tensor.data_ptr(),
                tensor.numel(),
                BLOCK_SIZE
            )
        else:
            output_tensor = tensor.clone()
        
        broadcasted_tensors.append(output_tensor)
    
    return broadcasted_tensors
