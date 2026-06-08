import torch
import triton
import triton.language as tl
from typing import Union, Tuple

@triton.jit
def _sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr, 
    output_ptr,
    input_height, 
    input_width,
    output_height, 
    output_width,
    input_batch,
    input_channels,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # Get the batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate the starting position for this block
    output_h_start = tl.program_id(2) * BLOCK_H
    output_w_start = tl.program_id(3) * BLOCK_W
    
    # Calculate the actual output dimensions for this block
    output_h_end = tl.minimum(output_h_start + BLOCK_H, output_height)
    output_w_end = tl.minimum(output_w_start + BLOCK_W, output_width)
    
    # Loop over the output dimensions
    for output_h in range(output_h_start, output_h_end):
        for output_w in range(output_w_start, output_w_end):
            # Calculate the input region boundaries
            h_start = (output_h * input_height) // output_height
            h_end = ((output_h + 1) * input_height + output_height - 1) // output_height
            w_start = (output_w * input_width) // output_width
            w_end = ((output_w + 1) * input_width + output_width - 1) // output_width
            
            # Calculate the number of elements in the region
            num_elements = (h_end - h_start) * (w_end - w_start)
            
            # Initialize sum
            sum_val = 0.0
            
            # Accumulate the values in the region
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    # Calculate the input index
                    input_idx = batch_idx * (input_height * input_width * input_channels) + \
                                channel_idx * (input_height * input_width) + \
                                h * input_width + w
                    sum_val += tl.load(input_ptr + input_idx)
            
            # Calculate the average
            avg_val = sum_val / num_elements
            
            # Apply sigmoid
            sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
            
            # Calculate the output index
            output_idx = batch_idx * (output_height * output_width * input_channels) + \
                         channel_idx * (output_height * output_width) + \
                         output_h * output_width + output_w
            
            # Store the result
            tl.store(output_ptr + output_idx, sigmoid_val)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    # Handle the case where output_size is an integer
    if isinstance(output_size, int):
        output_height = output_size
        output_width = output_size
    else:
        output_height, output_width = output_size
    
    # Get input dimensions
    input_batch, input_channels, input_height, input_width = input.shape
    
    # Create output tensor
    output = torch.empty(input_batch, input_channels, output_height, output_width, device=input.device, dtype=input.dtype)
    
    # Define block size
    BLOCK_H = 16
    BLOCK_W = 16
    
    # Launch kernel
    grid = (
        input_batch,      # batch dimension
        input_channels,   # channel dimension
        triton.cdiv(output_height, BLOCK_H),  # height blocks
        triton.cdiv(output_width, BLOCK_W)    # width blocks
    )
    
    _sigmoid_adaptive_avg_pool2d_kernel[grid](
        input,
        output,
        input_height,
        input_width,
        output_height,
        output_width,
        input_batch,
        input_channels,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return output
