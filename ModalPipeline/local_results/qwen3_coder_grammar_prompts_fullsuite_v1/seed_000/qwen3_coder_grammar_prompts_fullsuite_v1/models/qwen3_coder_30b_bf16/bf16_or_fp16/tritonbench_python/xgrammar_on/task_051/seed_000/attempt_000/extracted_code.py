import torch
import triton
import triton.language as tl

@triton.jit
def cos_avg_pool1d_kernel(
    input_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2,
    output_stride_0, output_stride_1, output_stride_2,
    input_size_0, input_size_1, input_size_2,
    output_size_2,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    ceil_mode: tl.constexpr,
    count_include_pad: tl.constexpr
):
    # Get the block index
    block_id = tl.program_id(0)
    
    # Calculate output indices
    output_batch = block_id // (input_size_1 * output_size_2)
    output_channel = (block_id % (input_size_1 * output_size_2)) // output_size_2
    output_pos = (block_id % (input_size_1 * output_size_2)) % output_size_2
    
    # Calculate input start position
    input_start = output_pos * stride - padding
    
    # Handle ceil mode
    if ceil_mode:
        input_end = input_start + kernel_size
    else:
        input_end = min(input_start + kernel_size, input_size_2)
    
    # Initialize sum and count
    sum_val = 0.0
    count = 0
    
    # Iterate over kernel
    for i in range(input_start, input_end):
        if i >= 0 and i < input_size_2:
            # Load input value
            input_val = tl.load(input_ptr + output_batch * input_stride_0 + output_channel * input_stride_1 + i * input_stride_2)
            # Apply cosine
            cos_val = tl.cos(input_val)
            sum_val += cos_val
            count += 1
        elif count_include_pad:
            # Include padding in average
            count += 1
    
    # Compute average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Store result
    tl.store(output_ptr + output_batch * output_stride_0 + output_channel * output_stride_1 + output_pos * output_stride_2, avg_val)


def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    # Handle default stride
    if stride is None:
        stride = kernel_size
    
    # Input tensor shape: (minibatch, in_channels, iW)
    assert input.dim() == 3, "Input tensor must be 3-dimensional"
    
    # Calculate output size
    batch_size, channels, input_size = input.shape
    
    # Compute output size
    if ceil_mode:
        output_size = (input_size + 2 * padding - kernel_size) // stride + 1
        if (input_size + 2 * padding - kernel_size) % stride != 0:
            output_size += 1
    else:
        output_size = (input_size + 2 * padding - kernel_size) // stride + 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_size, dtype=torch.float32, device=input.device)
    
    # Launch kernel
    grid = (batch_size * channels * output_size,)
    
    # Define block size
    block_size = 1024
    
    # Launch kernel
    cos_avg_pool1d_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        input_stride_0=input.stride(0),
        input_stride_1=input.stride(1),
        input_stride_2=input.stride(2),
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        output_stride_2=output.stride(2),
        input_size_0=input.size(0),
        input_size_1=input.size(1),
        input_size_2=input.size(2),
        output_size_2=output.size(2),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad
    )
    
    return output