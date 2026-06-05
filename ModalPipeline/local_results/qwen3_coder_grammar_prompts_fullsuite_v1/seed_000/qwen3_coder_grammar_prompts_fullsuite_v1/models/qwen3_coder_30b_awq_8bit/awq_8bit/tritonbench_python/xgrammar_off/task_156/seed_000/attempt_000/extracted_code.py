import torch
import triton
import triton.language as tl

@triton.jit
def ifftshift_kernel(input_ptr, output_ptr, n, dim_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    mask = indices < n
    input_data = tl.load(input_ptr + indices, mask=mask)
    
    # Compute shifted indices
    shifted_indices = tl.where(indices < dim_size // 2, 
                              indices + dim_size // 2, 
                              indices - dim_size // 2)
    
    # Store output data
    tl.store(output_ptr + shifted_indices, input_data, mask=mask)

def ifftshift(input, dim=None):
    if dim is None:
        # Apply ifftshift to all dimensions
        result = input.clone()
        for i in range(input.dim()):
            dim_size = input.shape[i]
            if dim_size > 1:
                # Create output tensor
                output = torch.empty_like(result)
                
                # Launch kernel for this dimension
                grid = (triton.cdiv(dim_size, 1024),)
                ifftshift_kernel[grid](
                    result.data_ptr() + result.stride(i) * 0,
                    output.data_ptr() + output.stride(i) * 0,
                    dim_size,
                    dim_size,
                    BLOCK_SIZE=1024
                )
                result = output
        return result
    else:
        # Apply ifftshift to specified dimensions
        if isinstance(dim, int):
            dim = [dim]
        
        result = input.clone()
        for d in dim:
            dim_size = input.shape[d]
            if dim_size > 1:
                # Create output tensor
                output = torch.empty_like(result)
                
                # Launch kernel for this dimension
                grid = (triton.cdiv(dim_size, 1024),)
                ifftshift_kernel[grid](
                    result.data_ptr() + result.stride(d) * 0,
                    output.data_ptr() + output.stride(d) * 0,
                    dim_size,
                    dim_size,
                    BLOCK_SIZE=1024
                )
                result = output
        return result
