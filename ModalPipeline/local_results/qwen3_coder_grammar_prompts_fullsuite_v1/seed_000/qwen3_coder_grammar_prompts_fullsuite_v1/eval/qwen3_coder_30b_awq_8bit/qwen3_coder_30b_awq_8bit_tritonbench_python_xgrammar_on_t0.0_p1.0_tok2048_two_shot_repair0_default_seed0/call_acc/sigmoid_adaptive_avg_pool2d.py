import torch
import triton
import triton.language as tl
from typing import Union, Tuple

@triton.jit
def _adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    input_height, input_width,
    output_height, output_width,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate the start and end indices for the input region
    h_start = (pid_h * input_height) // output_height
    h_end = ((pid_h + 1) * input_height + output_height - 1) // output_height
    w_start = (pid_w * input_width) // output_width
    w_end = ((pid_w + 1) * input_width + output_width - 1) // output_width
    
    # Calculate the number of elements in the region
    region_size = (h_end - h_start) * (w_end - w_start)
    
    # Load input values in the region
    sum_val = 0.0
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            input_idx = h * input_stride_0 + w * input_stride_1
            sum_val += tl.load(input_ptr + input_idx)
    
    # Calculate average
    avg_val = sum_val / region_size
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
    
    # Store result
    output_idx = pid_h * output_stride_0 + pid_w * output_stride_1
    tl.store(output_ptr + output_idx, sigmoid_val)

@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


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
    if len(input.shape) == 4:
        batch_size = input.shape[0]
        out = torch.empty((batch_size, input.shape[1], output_height, output_width), dtype=input.dtype, device=input.device)
        # Process each batch
        for b in range(batch_size):
            for c in range(input.shape[1]):
                # Flatten the input and output for kernel processing
                input_flat = input[b, c].contiguous()
                out_flat = out[b, c].contiguous()
                
                # Launch kernel
                grid_h = triton.cdiv(output_height, 16)
                grid_w = triton.cdiv(output_width, 16)
                grid = (grid_h, grid_w)
                
                _adaptive_avg_pool2d_kernel[grid](
                    input_flat, out_flat,
                    input_height, input_width,
                    output_height, output_width,
                    input_flat.stride(0), input_flat.stride(1),
                    out_flat.stride(0), out_flat.stride(1),
                    BLOCK_H=16,
                    BLOCK_W=16
                )
    else:
        # For 3D input (no batch dimension)
        out = torch.empty((input.shape[0], output_height, output_width), dtype=input.dtype, device=input.device)
        
        # Flatten the input and output for kernel processing
        input_flat = input.contiguous()
        out_flat = out.contiguous()
        
        # Launch kernel
        grid_h = triton.cdiv(output_height, 16)
        grid_w = triton.cdiv(output_width, 16)
        grid = (grid_h, grid_w)
        
        _adaptive_avg_pool2d_kernel[grid](
            input_flat, out_flat,
            input_height, input_width,
            output_height, output_width,
            input_flat.stride(0), input_flat.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_H=16,
            BLOCK_W=16
        )
    
    # Apply sigmoid to the final output
    out = out.view(-1)
    out_sigmoid = torch.empty_like(out)
    n = out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _sigmoid_kernel[grid](out, out_sigmoid, n, BLOCK=block)
    return out_sigmoid.view(out.shape)
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
