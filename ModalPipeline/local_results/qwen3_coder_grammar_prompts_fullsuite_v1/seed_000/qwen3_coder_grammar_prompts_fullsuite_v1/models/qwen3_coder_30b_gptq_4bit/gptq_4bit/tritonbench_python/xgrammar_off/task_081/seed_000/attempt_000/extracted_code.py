import torch
import triton
import triton.language as tl
from typing import Union, Tuple

@triton.jit
def _adaptive_avg_pool2d_sigmoid_kernel(
    input_ptr, 
    output_ptr, 
    input_height, 
    input_width, 
    output_height, 
    output_width, 
    batch_size, 
    channels,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate which batch and channel this program is handling
    batch_id = pid // (output_height * output_width)
    channel_id = (pid % (output_height * output_width)) // (output_height * output_width)
    
    # Calculate the output indices
    output_h = (pid % (output_height * output_width)) // output_width
    output_w = (pid % (output_height * output_width)) % output_width
    
    # Calculate the input region for this output element
    start_h = (output_h * input_height) // output_height
    end_h = ((output_h + 1) * input_height + output_height - 1) // output_height
    start_w = (output_w * input_width) // output_width
    end_w = ((output_w + 1) * input_width + output_width - 1) // output_width
    
    # Calculate the number of elements in the region
    count = (end_h - start_h) * (end_w - start_w)
    
    # Initialize sum
    sum_val = 0.0
    
    # Iterate through the input region
    for h in range(start_h, end_h):
        for w in range(start_w, end_w):
            # Calculate input index
            input_idx = batch_id * (input_height * input_width * channels) + \
                        h * (input_width * channels) + \
                        w * channels + \
                        channel_id
            
            # Load input value
            val = tl.load(input_ptr + input_idx, mask=True)
            sum_val += val
    
    # Calculate average
    avg_val = sum_val / count
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
    
    # Store result
    output_idx = batch_id * (output_height * output_width * channels) + \
                 output_h * (output_width * channels) + \
                 output_w * channels + \
                 channel_id
    
    tl.store(output_ptr + output_idx, sigmoid_val)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    # Handle scalar output_size
    if isinstance(output_size, int):
        output_height = output_size
        output_width = output_size
    else:
        output_height, output_width = output_size
    
    # Get input dimensions
    batch_size, channels, input_height, input_width = input.shape
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, device=input.device, dtype=input.dtype)
    
    # Calculate grid size
    total_elements = batch_size * channels * output_height * output_width
    BLOCK_SIZE = 256
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    _adaptive_avg_pool2d_sigmoid_kernel[grid_size](
        input, 
        output, 
        input_height, 
        input_width, 
        output_height, 
        output_width, 
        batch_size, 
        channels,
        BLOCK_H=16,
        BLOCK_W=16
    )
    
    return output
