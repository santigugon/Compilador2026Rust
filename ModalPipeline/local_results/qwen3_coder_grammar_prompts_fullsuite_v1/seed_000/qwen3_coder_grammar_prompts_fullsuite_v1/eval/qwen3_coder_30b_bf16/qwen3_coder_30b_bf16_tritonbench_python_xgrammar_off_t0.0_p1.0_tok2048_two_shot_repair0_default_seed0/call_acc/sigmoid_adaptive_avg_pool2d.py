import torch
import triton
import triton.language as tl
from typing import Union, Tuple

@triton.jit
def _sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    input_batch,
    input_height,
    input_width,
    output_height,
    output_width,
    input_batch_stride,
    input_height_stride,
    input_width_stride,
    output_batch_stride,
    output_height_stride,
    output_width_stride,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    batch_id = tl.program_id(0)
    h_id = tl.program_id(1)
    w_id = tl.program_id(2)
    
    # Calculate input region for this output element
    h_start = h_id * (input_height // output_height)
    h_end = min((h_id + 1) * (input_height // output_height), input_height)
    w_start = w_id * (input_width // output_width)
    w_end = min((w_id + 1) * (input_width // output_width), input_width)
    
    # Calculate average
    sum_val = 0.0
    count = 0
    
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            input_offset = batch_id * input_batch_stride + h * input_height_stride + w * input_width_stride
            sum_val += tl.load(input_ptr + input_offset)
            count += 1
    
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
    
    # Store result
    output_offset = batch_id * output_batch_stride + h_id * output_height_stride + w_id * output_width_stride
    tl.store(output_ptr + output_offset, sigmoid_val)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    if not isinstance(output_size, tuple):
        output_size = (output_size, output_size)
    
    batch, channels, height, width = input.shape
    output_h, output_w = output_size
    
    # Create output tensor
    output = torch.empty(batch, channels, output_h, output_w, dtype=input.dtype, device=input.device)
    
    # Define block size
    BLOCK_H = 16
    BLOCK_W = 16
    
    # Grid dimensions
    grid = (
        batch,  # batch dimension
        triton.cdiv(output_h, BLOCK_H),  # height dimension
        triton.cdiv(output_w, BLOCK_W)   # width dimension
    )
    
    # Calculate strides
    input_batch_stride = input.stride(0)
    input_height_stride = input.stride(2)
    input_width_stride = input.stride(3)
    output_batch_stride = output.stride(0)
    output_height_stride = output.stride(2)
    output_width_stride = output.stride(3)
    
    # Launch kernel
    _sigmoid_adaptive_avg_pool2d_kernel[grid](
        input,
        output,
        batch,
        height,
        width,
        output_h,
        output_w,
        input_batch_stride,
        input_height_stride,
        input_width_stride,
        output_batch_stride,
        output_height_stride,
        output_width_stride,
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
