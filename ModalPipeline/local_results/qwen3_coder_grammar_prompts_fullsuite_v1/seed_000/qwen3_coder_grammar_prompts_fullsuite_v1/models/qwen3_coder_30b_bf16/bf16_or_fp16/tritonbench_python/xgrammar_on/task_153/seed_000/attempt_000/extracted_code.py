import torch
import triton
import triton.language as tl

def adaptive_avg_pool2d(input, output_size):
    """
    Apply 2D adaptive average pooling to input tensor.
    
    Args:
        input: Input tensor of shape (N, C, H_in, W_in) or (C, H_in, W_in)
        output_size: Target output size (int, tuple, or None)
    
    Returns:
        Pooled tensor of shape (N, C, S_0, S_1) or (C, S_0, S_1)
    """
    # Handle different input shapes
    if input.dim() == 3:
        # Unbatched input: (C, H_in, W_in)
        batched = False
        C, H_in, W_in = input.shape
        N = 1
    else:
        # Batched input: (N, C, H_in, W_in)
        batched = True
        N, C, H_in, W_in = input.shape
    
    # Handle output_size
    if isinstance(output_size, int):
        H_out, W_out = output_size, output_size
    elif isinstance(output_size, tuple) and len(output_size) == 2:
        H_out, W_out = output_size
    else:
        raise ValueError("output_size must be int or tuple of length 2")
    
    # Handle None values
    if H_out is None:
        H_out = H_in
    if W_out is None:
        W_out = W_in
    
    # Create output tensor
    if batched:
        output = torch.empty(N, C, H_out, W_out, device=input.device, dtype=input.dtype)
    else:
        output = torch.empty(C, H_out, W_out, device=input.device, dtype=input.dtype)
    
    # Launch kernel
    if H_out == 0 or W_out == 0:
        return output
    
    # Define block size
    BLOCK_SIZE = 16
    grid = (
        triton.cdiv(H_out, BLOCK_SIZE),
        triton.cdiv(W_out, BLOCK_SIZE),
        N * C
    )
    
    _adaptive_avg_pool2d_kernel[grid](
        input, output,
        H_in, W_in, H_out, W_out,
        input.stride(0) if batched else 0,
        input.stride(1) if batched else input.stride(0),
        input.stride(2) if batched else input.stride(1),
        input.stride(3) if batched else input.stride(2),
        output.stride(0) if batched else 0,
        output.stride(1) if batched else output.stride(0),
        output.stride(2) if batched else output.stride(1),
        output.stride(3) if batched else output.stride(2),
        BLOCK_SIZE
    )
    
    return output

@triton.jit
def _adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    H_in, W_in, H_out, W_out,
    input_s0, input_s1, input_s2, input_s3,
    output_s0, output_s1, output_s2, output_s3,
    BLOCK_SIZE: tl.constexpr
):
    # Get block indices
    h_block = tl.program_id(0)
    w_block = tl.program_id(1)
    batch_channel = tl.program_id(2)
    
    # Get batch and channel indices
    if H_out > 0 and W_out > 0:
        batch = batch_channel // H_out
        channel = batch_channel % H_out
    else:
        batch = 0
        channel = 0
    
    # Calculate output coordinates
    h_start = h_block * BLOCK_SIZE
    w_start = w_block * BLOCK_SIZE
    h_end = min(h_start + BLOCK_SIZE, H_out)
    w_end = min(w_start + BLOCK_SIZE, W_out)
    
    # Process each element in the block
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            # Calculate input region bounds
            h_start_in = (h * H_in) // H_out
            h_end_in = ((h + 1) * H_in + H_out - 1) // H_out
            w_start_in = (w * W_in) // W_out
            w_end_in = ((w + 1) * W_in + W_out - 1) // W_out
            
            # Calculate sum
            sum_val = 0.0
            count = 0
            
            for h_in in range(h_start_in, h_end_in):
                for w_in in range(w_start_in, w_end_in):
                    # Calculate input pointer
                    input_idx = batch * input_s0 + channel * input_s1 + h_in * input_s2 + w_in * input_s3
                    sum_val += tl.load(input_ptr + input_idx)
                    count += 1
            
            # Calculate average
            if count > 0:
                avg_val = sum_val / count
            else:
                avg_val = 0.0
            
            # Store result
            output_idx = batch * output_s0 + channel * output_s1 + h * output_s2 + w * output_s3
            tl.store(output_ptr + output_idx, avg_val)