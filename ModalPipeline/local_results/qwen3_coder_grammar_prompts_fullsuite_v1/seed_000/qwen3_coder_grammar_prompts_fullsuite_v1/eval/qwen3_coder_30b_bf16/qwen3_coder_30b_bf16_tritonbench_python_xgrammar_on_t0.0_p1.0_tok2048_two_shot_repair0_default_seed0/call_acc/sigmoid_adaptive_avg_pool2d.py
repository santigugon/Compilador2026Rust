import torch
import triton
import triton.language as tl
from typing import Union, Tuple

@triton.jit
def _sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    input_height, input_width,
    output_height, output_width,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Get the output coordinates
    output_y = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    output_x = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for valid output indices
    mask_y = output_y < output_height
    mask_x = output_x < output_width
    
    # Broadcast output coordinates to all elements in the block
    output_y = output_y[:, None]
    output_x = output_x[None, :]
    
    # Calculate the input region for each output element
    # For adaptive pooling, we need to map output coordinates to input coordinates
    start_h = (output_y * input_height) // output_height
    end_h = ((output_y + 1) * input_height + output_height - 1) // output_height
    start_w = (output_x * input_width) // output_width
    end_w = ((output_x + 1) * input_width + output_width - 1) // output_width
    
    # Calculate the average
    sum_val = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    count = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    
    # Loop over the input region
    for h in range(start_h.min(), end_h.max()):
        for w in range(start_w.min(), end_w.max()):
            # Check if this input position is within the valid range
            h_mask = (h >= start_h) & (h < end_h)
            w_mask = (w >= start_w) & (w < end_w)
            mask = h_mask & w_mask
            
            # Load input value
            input_val = tl.load(input_ptr + h * input_stride_0 + w * input_stride_1, mask=mask, other=0.0)
            
            # Accumulate sum and count
            sum_val += input_val * mask
            count += mask
    
    # Compute average
    avg = sum_val / (count + 1e-8)
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-avg))
    
    # Store output
    output_y = output_y[:, None]
    output_x = output_x[None, :]
    output_ptr = output_ptr + output_y * output_stride_0 + output_x * output_stride_1
    tl.store(output_ptr, sigmoid_val, mask=mask_y[:, None] & mask_x[None, :])

@triton.jit
def _sigmoid_adaptive_avg_pool2d_kernel_simple(
    input_ptr, output_ptr,
    input_height, input_width,
    output_height, output_width,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    BLOCK_SIZE: tl.constexpr
):
    # Get the output coordinates
    output_y = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_x = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid output indices
    mask_y = output_y < output_height
    mask_x = output_x < output_width
    
    # Broadcast output coordinates to all elements in the block
    output_y = output_y[:, None]
    output_x = output_x[None, :]
    
    # Calculate the input region for each output element
    start_h = (output_y * input_height) // output_height
    end_h = ((output_y + 1) * input_height + output_height - 1) // output_height
    start_w = (output_x * input_width) // output_width
    end_w = ((output_x + 1) * input_width + output_width - 1) // output_width
    
    # Calculate the average
    sum_val = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
    
    # Loop over the input region
    for h in range(start_h.min(), end_h.max()):
        for w in range(start_w.min(), end_w.max()):
            # Check if this input position is within the valid range
            h_mask = (h >= start_h) & (h < end_h)
            w_mask = (w >= start_w) & (w < end_w)
            mask = h_mask & w_mask
            
            # Load input value
            input_val = tl.load(input_ptr + h * input_stride_0 + w * input_stride_1, mask=mask, other=0.0)
            
            # Accumulate sum and count
            sum_val += input_val * mask
            count += mask
    
    # Compute average
    avg = sum_val / (count + 1e-8)
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-avg))
    
    # Store output
    output_y = output_y[:, None]
    output_x = output_x[None, :]
    output_ptr = output_ptr + output_y * output_stride_0 + output_x * output_stride_1
    tl.store(output_ptr, sigmoid_val, mask=mask_y[:, None] & mask_x[None, :])

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    # Handle scalar output_size
    if isinstance(output_size, int):
        output_height = output_size
        output_width = output_size
    else:
        output_height, output_width = output_size
    
    # Get input dimensions
    input_height, input_width = input.shape[-2], input.shape[-1]
    
    # Create output tensor
    if len(input.shape) == 4:
        batch_size, channels = input.shape[0], input.shape[1]
        output = torch.empty((batch_size, channels, output_height, output_width), dtype=torch.float32, device=input.device)
    else:
        output = torch.empty((output_height, output_width), dtype=torch.float32, device=input.device)
    
    # Get strides
    if len(input.shape) == 4:
        input_stride_0, input_stride_1 = input.stride(2), input.stride(3)
        output_stride_0, output_stride_1 = output.stride(2), output.stride(3)
    else:
        input_stride_0, input_stride_1 = input.stride(0), input.stride(1)
        output_stride_0, output_stride_1 = output.stride(0), output.stride(1)
    
    # Launch kernel
    if len(input.shape) == 4:
        batch_size, channels = input.shape[0], input.shape[1]
        for b in range(batch_size):
            for c in range(channels):
                input_ptr = input[b, c].data_ptr()
                output_ptr = output[b, c].data_ptr()
                grid = (
                    triton.cdiv(output_height, 16),
                    triton.cdiv(output_width, 16)
                )
                _sigmoid_adaptive_avg_pool2d_kernel_simple[grid](
                    input_ptr, output_ptr,
                    input_height, input_width,
                    output_height, output_width,
                    input_stride_0, input_stride_1,
                    output_stride_0, output_stride_1,
                    BLOCK_SIZE=16
                )
    else:
        grid = (
            triton.cdiv(output_height, 16),
            triton.cdiv(output_width, 16)
        )
        _sigmoid_adaptive_avg_pool2d_kernel_simple[grid](
            input.data_ptr(), output.data_ptr(),
            input_height, input_width,
            output_height, output_width,
            input_stride_0, input_stride_1,
            output_stride_0, output_stride_1,
            BLOCK_SIZE=16
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
