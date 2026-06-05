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
    input_stride_0, 
    input_stride_1,
    output_stride_0, 
    output_stride_1,
    BLOCK_H: tl.constexpr, 
    BLOCK_W: tl.constexpr
):
    # Get the output coordinates
    output_y = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    output_x = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for valid output indices
    mask_y = output_y < output_height
    mask_x = output_x < output_width
    mask = mask_y[:, None] & mask_x[None, :]
    
    # Calculate the input region for each output element
    # For adaptive pooling, we need to map output coordinates to input regions
    h_start = (output_y * input_height) // output_height
    h_end = ((output_y + 1) * input_height + output_height - 1) // output_height
    w_start = (output_x * input_width) // output_width
    w_end = ((output_x + 1) * input_width + output_width - 1) // output_width
    
    # Calculate the average
    sum_val = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    count = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    
    # Loop over the input region for each output element
    for h in range(h_start.min(), h_end.max()):
        for w in range(w_start.min(), w_end.max()):
            # Check if this input position is within the valid range
            h_in = tl.minimum(h, input_height - 1)
            w_in = tl.minimum(w, input_width - 1)
            
            # Load input value
            input_val = tl.load(input_ptr + h_in * input_stride_0 + w_in * input_stride_1, mask=False)
            
            # Check if this position is within the pooling region
            h_in_start = h_start
            h_in_end = h_end
            w_in_start = w_start
            w_in_end = w_end
            
            # Add to sum if within bounds
            in_mask = (h >= h_in_start) & (h < h_in_end) & (w >= w_in_start) & (w < w_in_end)
            sum_val += tl.where(in_mask, input_val, 0.0)
            count += tl.where(in_mask, 1.0, 0.0)
    
    # Compute average
    avg_val = tl.where(count > 0, sum_val / count, 0.0)
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
    
    # Store result
    output_ptr += output_y[:, None] * output_stride_0 + output_x[None, :] * output_stride_1
    tl.store(output_ptr, sigmoid_val, mask=mask)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    # Handle output_size
    if isinstance(output_size, int):
        output_height = output_size
        output_width = output_size
    else:
        output_height, output_width = output_size
    
    # Get input dimensions
    input_height, input_width = input.shape[-2], input.shape[-1]
    
    # Create output tensor
    output_shape = list(input.shape[:-2]) + [output_height, output_width]
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Get strides
    input_stride_0, input_stride_1 = input.stride(-2), input.stride(-1)
    output_stride_0, output_stride_1 = output.stride(-2), output.stride(-1)
    
    # Launch kernel
    BLOCK_H = 16
    BLOCK_W = 16
    
    grid_h = triton.cdiv(output_height, BLOCK_H)
    grid_w = triton.cdiv(output_width, BLOCK_W)
    grid = (grid_h, grid_w)
    
    _adaptive_avg_pool2d_sigmoid_kernel[grid](
        input, 
        output,
        input_height, 
        input_width,
        output_height, 
        output_width,
        input_stride_0, 
        input_stride_1,
        output_stride_0, 
        output_stride_1,
        BLOCK_H=BLOCK_H, 
        BLOCK_W=BLOCK_W
    )
    
    return output

##################################################################################################################################################



def test_sigmoid_adaptive_avg_pool2d():
    # Initialize a dictionary to store the results of each test case
    results = {}

    # Test case 1: Basic test with a 4D tensor and output size as an integer
    input_tensor1 = torch.randn(1, 3, 8, 8, device='cuda')  # Batch size 1, 3 channels, 8x8 size
    output_size1 = 4
    result1 = sigmoid_adaptive_avg_pool2d(input_tensor1, output_size1)
    results["test_case_1"] = result1

    # Test case 2: Test with a 4D tensor and output size as a tuple
    input_tensor2 = torch.randn(2, 3, 10, 10, device='cuda')  # Batch size 2, 3 channels, 10x10 size
    output_size2 = (5, 5)
    result2 = sigmoid_adaptive_avg_pool2d(input_tensor2, output_size2)
    results["test_case_2"] = result2

    # Test case 3: Test with a larger batch size
    input_tensor3 = torch.randn(4, 3, 16, 16, device='cuda')  # Batch size 4, 3 channels, 16x16 size
    output_size3 = (8, 8)
    result3 = sigmoid_adaptive_avg_pool2d(input_tensor3, output_size3)
    results["test_case_3"] = result3

    # Test case 4: Test with a single channel
    input_tensor4 = torch.randn(1, 1, 12, 12, device='cuda')  # Batch size 1, 1 channel, 12x12 size
    output_size4 = (6, 6)
    result4 = sigmoid_adaptive_avg_pool2d(input_tensor4, output_size4)
    results["test_case_4"] = result4

    return results

test_results = test_sigmoid_adaptive_avg_pool2d()
