import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr, 
    input_row_stride, input_col_stride,
    output_row_stride, output_col_stride,
    output_height, output_width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr
):
    # Get the thread index
    output_row = tl.program_id(0)
    output_col = tl.program_id(1)
    
    # Check bounds
    if output_row >= output_height or output_col >= output_width:
        return
    
    # Calculate the input region for this output element
    input_start_h = (output_row * input_row_stride) // output_height
    input_end_h = ((output_row + 1) * input_row_stride) // output_height
    input_start_w = (output_col * input_col_stride) // output_width
    input_end_w = ((output_col + 1) * input_col_stride) // output_width
    
    # Calculate the average
    sum_val = 0.0
    count = 0
    
    for h in range(input_start_h, input_end_h):
        for w in range(input_start_w, input_end_w):
            sum_val += tl.load(input_ptr + h * input_row_stride + w * input_col_stride)
            count += 1
    
    # Compute average
    avg_val = sum_val / count if count > 0 else 0.0
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
    
    # Write result
    output_idx = output_row * output_row_stride + output_col * output_col_stride
    tl.store(output_ptr + output_idx, sigmoid_val)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
    # Ensure input is on GPU
    if input.device.type != 'cuda':
        input = input.cuda()
    
    # Handle output_size as int
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    # Get input dimensions
    batch_size, channels, input_height, input_width = input.shape
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_size[0], output_size[1], device=input.device, dtype=torch.float32)
    
    # Define block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    # Launch kernel
    grid = (
        triton.cdiv(output_size[0], BLOCK_SIZE_H),
        triton.cdiv(output_size[1], BLOCK_SIZE_W)
    )
    
    # Calculate strides
    input_row_stride = input.stride(2)
    input_col_stride = input.stride(3)
    output_row_stride = output.stride(2)
    output_col_stride = output.stride(3)
    
    # Launch kernel
    sigmoid_adaptive_avg_pool2d_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        input_row_stride=input_row_stride,
        input_col_stride=input_col_stride,
        output_row_stride=output_row_stride,
        output_col_stride=output_col_stride,
        output_height=output_size[0],
        output_width=output_size[1],
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return output
