import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_log_kernel(input_ptr, output_ptr, dim_size: tl.constexpr, n_elements: tl.constexpr, 
                       stride_input, stride_output, dim_stride, BLOCK: tl.constexpr):
    # Get the program ID for the dimension we're processing
    pid = tl.program_id(0)
    
    # Calculate the starting offset for this block
    block_start = pid * BLOCK
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets * stride_input, mask=mask, other=0.0)
    
    # Apply log operation
    log_data = tl.log(input_data)
    
    # Apply softmax along the specified dimension
    # For softmax, we need to:
    # 1. Subtract the max for numerical stability
    # 2. Apply exp
    # 3. Sum and normalize
    
    # Find max value for numerical stability
    max_val = tl.max(log_data, axis=0)
    
    # Subtract max for numerical stability
    shifted = log_data - max_val
    
    # Apply exp
    exp_val = tl.exp(shifted)
    
    # Sum along the dimension
    sum_val = tl.sum(exp_val, axis=0)
    
    # Normalize
    softmax_val = exp_val / sum_val
    
    # Store result
    tl.store(output_ptr + offsets * stride_output, softmax_val, mask=mask)

def softmax_log(input, dim=-1, dtype=None):
    # Handle dtype casting if specified
    if dtype is not None:
        input = input.to(dtype)
    
    # Ensure input is contiguous for easier handling
    input = input.contiguous()
    
    # Get output tensor with same shape as input
    output = torch.empty_like(input)
    
    # Get the size of the specified dimension
    dim_size = input.size(dim)
    
    # Get total number of elements
    n_elements = input.numel()
    
    # Calculate strides
    input_stride = input.stride(dim) if input.dim() > 0 else 1
    output_stride = output.stride(dim) if output.dim() > 0 else 1
    
    # Set block size
    BLOCK = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK),)
    
    # Launch kernel
    _softmax_log_kernel[grid](
        input,
        output,
        dim_size,
        n_elements,
        input_stride,
        output_stride,
        input.stride(dim) if input.dim() > 0 else 1,
        BLOCK=BLOCK
    )
    
    return output
