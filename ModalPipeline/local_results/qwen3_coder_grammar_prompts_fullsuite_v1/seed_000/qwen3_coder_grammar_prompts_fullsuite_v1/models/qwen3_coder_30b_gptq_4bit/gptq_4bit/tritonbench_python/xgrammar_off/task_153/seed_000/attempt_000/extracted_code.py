import torch
import triton
import triton.language as tl

@triton.jit
def _adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    n_elements, H_out: tl.constexpr, W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate output indices
    output_idx = tl.load(offsets, mask=mask, other=0)
    batch_idx = output_idx // (H_out * W_out)
    spatial_idx = output_idx % (H_out * W_out)
    h_out = spatial_idx // W_out
    w_out = spatial_idx % W_out
    
    # Calculate input ranges
    h_in_start = (h_out * input_stride_2) // H_out
    h_in_end = ((h_out + 1) * input_stride_2) // H_out
    w_in_start = (w_out * input_stride_3) // W_out
    w_in_end = ((w_in_start + 1) * input_stride_3) // W_out
    
    # Compute average
    sum_val = 0.0
    count = 0
    for h_in in range(h_in_start, h_in_end):
        for w_in in range(w_in_start, w_in_end):
            input_offset = batch_idx * input_stride_0 + h_in * input_stride_2 + w_in * input_stride_3
            sum_val += tl.load(input_ptr + input_offset, mask=True)
            count += 1
    
    # Store result
    output_offset = batch_idx * output_stride_0 + h_out * output_stride_2 + w_out * output_stride_3
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    tl.store(output_ptr + output_offset, avg_val, mask=True)

def adaptive_avg_pool2d(input, output_size):
    # Handle input shape and output size
    if len(input.shape) == 3:
        # (C, H_in, W_in)
        batch_size = 1
        channels = input.shape[0]
        h_in = input.shape[1]
        w_in = input.shape[2]
        input_reshaped = input
    else:
        # (N, C, H_in, W_in)
        batch_size = input.shape[0]
        channels = input.shape[1]
        h_in = input.shape[2]
        w_in = input.shape[3]
        input_reshaped = input
    
    # Process output_size
    if isinstance(output_size, int):
        h_out = output_size
        w_out = output_size
    elif isinstance(output_size, tuple) and len(output_size) == 2:
        h_out = output_size[0]
        w_out = output_size[1]
    else:
        raise ValueError("output_size must be an int or a tuple of two ints")
    
    # Create output tensor
    if len(input.shape) == 3:
        output = torch.empty(channels, h_out, w_out, device=input.device, dtype=input.dtype)
    else:
        output = torch.empty(batch_size, channels, h_out, w_out, device=input.device, dtype=input.dtype)
    
    # Calculate total elements
    total_elements = batch_size * channels * h_out * w_out
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Get strides
    if len(input.shape) == 3:
        input_stride_0 = input.stride(0)  # C
        input_stride_1 = input.stride(1)  # H_in
        input_stride_2 = input.stride(2)  # W_in
        input_stride_3 = 1  # For indexing
        output_stride_0 = output.stride(0)  # C
        output_stride_1 = output.stride(1)  # H_out
        output_stride_2 = output.stride(2)  # W_out
        output_stride_3 = 1  # For indexing
    else:
        input_stride_0 = input.stride(0)  # N
        input_stride_1 = input.stride(1)  # C
        input_stride_2 = input.stride(2)  # H_in
        input_stride_3 = input.stride(3)  # W_in
        output_stride_0 = output.stride(0)  # N
        output_stride_1 = output.stride(1)  # C
        output_stride_2 = output.stride(2)  # H_out
        output_stride_3 = output.stride(3)  # W_out
    
    # Launch kernel
    _adaptive_avg_pool2d_kernel[grid](
        input_reshaped,
        output,
        input_stride_0, input_stride_1, input_stride_2, input_stride_3,
        output_stride_0, output_stride_1, output_stride_2, output_stride_3,
        total_elements,
        h_out,
        w_out,
        BLOCK_SIZE
    )
    
    return output
