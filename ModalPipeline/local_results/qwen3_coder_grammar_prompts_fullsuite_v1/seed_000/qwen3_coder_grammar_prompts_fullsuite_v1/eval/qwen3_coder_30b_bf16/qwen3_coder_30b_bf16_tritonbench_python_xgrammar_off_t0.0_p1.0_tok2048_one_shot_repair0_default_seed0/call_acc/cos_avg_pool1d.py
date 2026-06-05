import torch
import triton
import triton.language as tl

@triton.jit
def cos_avg_pool1d_kernel(
    input_ptr, output_ptr, 
    input_stride_0, input_stride_1, input_stride_2,
    output_stride_0, output_stride_1, output_stride_2,
    N, 
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    ceil_mode: tl.constexpr,
    count_include_pad: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    output_idx = tl.program_id(2)
    
    # Calculate the starting position in the input tensor
    input_start = output_idx * stride - padding
    input_end = input_start + kernel_size
    
    # Handle ceil mode
    if ceil_mode:
        output_size = (N + 2 * padding - kernel_size + stride - 1) // stride + 1
    else:
        output_size = (N + 2 * padding - kernel_size) // stride + 1
    
    # Ensure we don't go out of bounds
    input_start = tl.max(input_start, 0)
    input_end = tl.min(input_end, N + 2 * padding)
    
    # Calculate the number of elements to process
    num_elements = input_end - input_start
    
    # Initialize accumulator
    sum_val = 0.0
    count = 0
    
    # Loop over the kernel window
    for i in range(input_start, input_end):
        # Check if we're within the valid input range
        if i >= 0 and i < N:
            # Load input value
            input_offset = batch_idx * input_stride_0 + channel_idx * input_stride_1 + i * input_stride_2
            input_val = tl.load(input_ptr + input_offset)
            # Apply cosine
            input_val = tl.cos(input_val)
            # Accumulate
            sum_val += input_val
            count += 1
        elif count_include_pad:
            # Include padding in average
            sum_val += 0.0
            count += 1
    
    # Compute average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Store result
    output_offset = batch_idx * output_stride_0 + channel_idx * output_stride_1 + output_idx * output_stride_2
    tl.store(output_ptr + output_offset, avg_val)

def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    # Ensure input is on GPU
    if input.device.type != 'cuda':
        raise ValueError("Input tensor must be on CUDA device")
    
    # Set default stride
    if stride is None:
        stride = kernel_size
    
    # Get input dimensions
    batch_size, channels, input_length = input.shape
    
    # Calculate output length
    if ceil_mode:
        output_length = (input_length + 2 * padding - kernel_size + stride - 1) // stride + 1
    else:
        output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_length, device=input.device, dtype=input.dtype)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = (
        batch_size,
        channels,
        output_length
    )
    
    # Get strides
    input_stride_0, input_stride_1, input_stride_2 = input.stride()
    output_stride_0, output_stride_1, output_stride_2 = output.stride()
    
    # Launch kernel
    cos_avg_pool1d_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        input_stride_0=input_stride_0,
        input_stride_1=input_stride_1,
        input_stride_2=input_stride_2,
        output_stride_0=output_stride_0,
        output_stride_1=output_stride_1,
        output_stride_2=output_stride_2,
        N=input_length,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int=None, padding: int=0, ceil_mode: bool=False, count_include_pad: bool=True) -> torch.Tensor:
#     cos_input = torch.cos(input)
#     return F.avg_pool1d(cos_input, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)

def test_cos_avg_pool1d():
    results = {}

    # Test case 1: Basic functionality with default parameters
    input_tensor_1 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_1'] = cos_avg_pool1d(input_tensor_1, kernel_size=2)

    # Test case 2: Custom stride
    input_tensor_2 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_2'] = cos_avg_pool1d(input_tensor_2, kernel_size=2, stride=1)

    # Test case 3: With padding
    input_tensor_3 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_3'] = cos_avg_pool1d(input_tensor_3, kernel_size=2, padding=1)

    # Test case 4: Using ceil_mode
    input_tensor_4 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_4'] = cos_avg_pool1d(input_tensor_4, kernel_size=2, ceil_mode=True)

    return results

test_results = test_cos_avg_pool1d()
