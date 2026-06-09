import torch
import triton
import triton.language as tl

def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    if stride is None:
        stride = kernel_size
    
    # Get input dimensions
    batch_size, channels, input_width = input.shape
    
    # Calculate output width
    if ceil_mode:
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    else:
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    
    # Adjust output width for ceil_mode
    if ceil_mode and (input_width + 2 * padding - kernel_size) % stride != 0:
        output_width += 1
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_width, device=input.device, dtype=input.dtype)
    
    # Handle the case where kernel_size is 1 and stride is 1
    if kernel_size == 1 and stride == 1 and padding == 0:
        # Just apply cosine
        return torch.cos(input)
    
    # Apply cosine first
    cos_input = torch.cos(input)
    
    # Prepare for pooling
    # We'll use a two-step approach: first compute cosine, then pool
    # For pooling, we'll use a kernel that handles the averaging
    
    # Define kernel
    @triton.jit
    def _cos_avg_pool1d_kernel(
        input_ptr, output_ptr,
        input_batch_stride, input_channel_stride, input_width_stride,
        output_batch_stride, output_channel_stride, output_width_stride,
        input_width, output_width,
        kernel_size: tl.constexpr,
        stride: tl.constexpr,
        padding: tl.constexpr,
        count_include_pad: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        batch_idx = tl.program_id(0)
        channel_idx = tl.program_id(1)
        output_idx = tl.program_id(2)
        
        # Calculate the starting position in the input
        start_pos = output_idx * stride - padding
        
        # Initialize sum
        sum_val = 0.0
        count = 0
        
        # Loop over kernel
        for i in range(kernel_size):
            input_pos = start_pos + i
            # Check bounds
            if input_pos >= 0 and input_pos < input_width:
                # Load input value
                input_offset = batch_idx * input_batch_stride + channel_idx * input_channel_stride + input_pos * input_width_stride
                val = tl.load(input_ptr + input_offset, mask=(input_pos >= 0) & (input_pos < input_width), other=0.0)
                sum_val += val
                count += 1
            elif count_include_pad:
                # If padding is included, we still count the padding
                count += 1
        
        # Calculate average
        if count > 0:
            avg_val = sum_val / count
        else:
            avg_val = 0.0
        
        # Store result
        output_offset = batch_idx * output_batch_stride + channel_idx * output_channel_stride + output_idx * output_width_stride
        tl.store(output_ptr + output_offset, avg_val)
    
    # Launch kernel
    grid = (batch_size, channels, output_width)
    
    # Calculate strides
    input_batch_stride = input.stride(0)
    input_channel_stride = input.stride(1)
    input_width_stride = input.stride(2)
    
    output_batch_stride = output.stride(0)
    output_channel_stride = output.stride(1)
    output_width_stride = output.stride(2)
    
    # Launch kernel
    _cos_avg_pool1d_kernel[grid](
        cos_input,
        output,
        input_batch_stride,
        input_channel_stride,
        input_width_stride,
        output_batch_stride,
        output_channel_stride,
        output_width_stride,
        input_width,
        output_width,
        kernel_size,
        stride,
        padding,
        count_include_pad,
        BLOCK_SIZE=128
    )
    
    return output