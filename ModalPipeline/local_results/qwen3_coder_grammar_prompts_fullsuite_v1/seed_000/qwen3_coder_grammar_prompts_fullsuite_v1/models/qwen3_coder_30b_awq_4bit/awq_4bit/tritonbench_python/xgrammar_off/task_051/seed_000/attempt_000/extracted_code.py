import torch
import triton
import triton.language as tl

@triton.jit
def cos_avg_pool1d_kernel(
    input_ptr, 
    output_ptr, 
    input_size, 
    output_size, 
    kernel_size, 
    stride, 
    padding, 
    count_include_pad,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_pos = block_idx * BLOCK_SIZE
    
    # Loop over the output elements
    for i in range(0, BLOCK_SIZE):
        if start_pos + i >= output_size:
            break
            
        # Calculate the pooling window start and end positions
        pool_start = start_pos + i * stride - padding
        pool_end = pool_start + kernel_size
        
        # Adjust for padding
        pool_start = tl.max(pool_start, 0)
        pool_end = tl.min(pool_end, input_size)
        
        # Calculate the number of elements in the pooling window
        num_elements = pool_end - pool_start
        
        # If no elements, set output to 0
        if num_elements <= 0:
            tl.store(output_ptr + start_pos + i, 0.0)
            continue
            
        # Compute the sum of cosine values in the pooling window
        sum_cos = 0.0
        for j in range(pool_start, pool_end):
            input_val = tl.load(input_ptr + j)
            sum_cos += tl.cos(input_val)
            
        # Compute the average
        avg = sum_cos / num_elements if not count_include_pad or num_elements > 0 else 0.0
        
        # Store the result
        tl.store(output_ptr + start_pos + i, avg)

def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    # Handle default stride
    if stride is None:
        stride = kernel_size
    
    # Get input dimensions
    batch_size, channels, input_length = input.shape
    
    # Calculate output length
    if ceil_mode:
        output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    else:
        output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_length, device=input.device, dtype=input.dtype)
    
    # Flatten input for easier processing
    input_flat = input.view(-1, input_length)
    output_flat = output.view(-1, output_length)
    
    # Launch kernel
    for i in range(input_flat.shape[0]):
        # Get input and output pointers
        input_ptr = input_flat[i].data_ptr()
        output_ptr = output_flat[i].data_ptr()
        
        # Launch kernel with appropriate grid size
        grid_size = (output_length + 256 - 1) // 256
        cos_avg_pool1d_kernel[grid_size](
            input_ptr,
            output_ptr,
            input_length,
            output_length,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            BLOCK_SIZE=256
        )
    
    return output
