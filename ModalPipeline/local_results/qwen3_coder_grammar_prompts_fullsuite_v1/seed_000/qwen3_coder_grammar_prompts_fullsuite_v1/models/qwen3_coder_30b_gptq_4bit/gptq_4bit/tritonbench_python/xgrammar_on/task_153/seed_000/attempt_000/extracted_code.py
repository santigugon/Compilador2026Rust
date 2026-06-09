import torch
import triton
import triton.language as tl

def adaptive_avg_pool2d(input, output_size):
    # Handle input shape and determine batch dimension
    if input.dim() == 4:
        N, C, H_in, W_in = input.shape
        batched = True
    elif input.dim() == 3:
        C, H_in, W_in = input.shape
        N = 1
        batched = False
    else:
        raise ValueError("Input must be 3D or 4D")
    
    # Process output_size
    if isinstance(output_size, (int, float)):
        H_out, W_out = int(output_size), int(output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        H_out, W_out = int(output_size[0]), int(output_size[1])
    else:
        raise ValueError("output_size must be an int or a tuple of two ints")
    
    # Handle None values in output_size
    if H_out is None:
        H_out = H_in
    if W_out is None:
        W_out = W_in
    
    # Create output tensor
    if batched:
        output = torch.empty(N, C, H_out, W_out, device=input.device, dtype=input.dtype)
    else:
        output = torch.empty(C, H_out, W_out, device=input.device, dtype=input.dtype)
    
    # Define block size
    BLOCK = 256
    
    # Launch kernel
    grid = (triton.cdiv(H_out * W_out, BLOCK),)
    _adaptive_avg_pool2d_kernel[grid](input, output, N, C, H_in, W_in, H_out, W_out, input.stride(0), input.stride(1), input.stride(2), input.stride(3), output.stride(0), output.stride(1), output.stride(2), output.stride(3), BLOCK=BLOCK)
    
    return output

@triton.jit
def _adaptive_avg_pool2d_kernel(input_ptr, output_ptr, N, C, H_in, W_in, H_out, W_out, input_stride_0, input_stride_1, input_stride_2, input_stride_3, output_stride_0, output_stride_1, output_stride_2, output_stride_3, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate output indices
    output_idx = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Calculate total output elements
    total_output_elements = N * C * H_out * W_out
    
    # Mask for valid indices
    mask = output_idx < total_output_elements
    
    # Calculate which output element we're processing
    output_idx = tl.where(mask, output_idx, 0)
    
    # Calculate batch, channel, height, and width indices
    batch_idx = output_idx // (C * H_out * W_out)
    remaining = output_idx % (C * H_out * W_out)
    channel_idx = remaining // (H_out * W_out)
    remaining = remaining % (H_out * W_out)
    h_idx = remaining // W_out
    w_idx = remaining % W_out
    
    # Calculate input region bounds
    h_start = (h_idx * H_in) // H_out
    h_end = ((h_idx + 1) * H_in + H_out - 1) // H_out
    w_start = (w_idx * W_in) // W_out
    w_end = ((w_idx + 1) * W_in + W_out - 1) // W_out
    
    # Calculate average
    sum_val = 0.0
    count = 0
    
    # Iterate through input region
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            # Calculate input index
            input_idx = batch_idx * input_stride_0 + channel_idx * input_stride_1 + h * input_stride_2 + w * input_stride_3
            sum_val += tl.load(input_ptr + input_idx, mask=True)
            count += 1
    
    # Calculate average
    avg_val = sum_val / count if count > 0 else 0.0
    
    # Store result
    output_idx = batch_idx * output_stride_0 + channel_idx * output_stride_1 + h_idx * output_stride_2 + w_idx * output_stride_3
    tl.store(output_ptr + output_idx, avg_val, mask=True)