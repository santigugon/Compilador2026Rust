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
    # Handle input shape
    if input.dim() == 3:
        # Unbatched case: (C, H_in, W_in)
        batched = False
        C, H_in, W_in = input.shape
        N = 1
    else:
        # Batched case: (N, C, H_in, W_in)
        batched = True
        N, C, H_in, W_in = input.shape
    
    # Handle output size
    if output_size is None:
        # Use input size
        S_0, S_1 = H_in, W_in
    elif isinstance(output_size, int):
        # Square output
        S_0, S_1 = output_size, output_size
    else:
        # Tuple output
        S_0, S_1 = output_size
    
    # Create output tensor
    if batched:
        output = torch.empty(N, C, S_0, S_1, device=input.device, dtype=input.dtype)
    else:
        output = torch.empty(C, S_0, S_1, device=input.device, dtype=input.dtype)
    
    # Handle case where output size is same as input
    if S_0 == H_in and S_1 == W_in:
        return input.clone()
    
    # Launch kernel
    if batched:
        _adaptive_avg_pool2d_kernel[(N, C, S_0, S_1)](input, output, H_in, W_in, S_0, S_1)
    else:
        _adaptive_avg_pool2d_kernel[(C, S_0, S_1)](input, output, H_in, W_in, S_0, S_1)
    
    return output

@triton.jit
def _adaptive_avg_pool2d_kernel(input_ptr, output_ptr, H_in: tl.constexpr, W_in: tl.constexpr, S_0: tl.constexpr, S_1: tl.constexpr):
    # Get thread indices
    if S_0 > 1:
        batch_idx = tl.program_id(0)
        channel_idx = tl.program_id(1)
        h_out_idx = tl.program_id(2)
    else:
        batch_idx = tl.program_id(0)
        channel_idx = tl.program_id(1)
        h_out_idx = 0
    
    if S_1 > 1:
        w_out_idx = tl.program_id(3)
    else:
        w_out_idx = 0
    
    # Calculate input region boundaries
    h_start = (h_out_idx * H_in) // S_0
    h_end = ((h_out_idx + 1) * H_in + S_0 - 1) // S_0
    w_start = (w_out_idx * W_in) // S_1
    w_end = ((w_out_idx + 1) * W_in + S_1 - 1) // S_1
    
    # Calculate sum
    sum_val = 0.0
    count = 0
    
    # Iterate through input region
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            if S_0 > 1:
                input_idx = batch_idx * (C * H_in * W_in) + channel_idx * (H_in * W_in) + h * W_in + w
            else:
                input_idx = channel_idx * (H_in * W_in) + h * W_in + w
            
            sum_val += tl.load(input_ptr + input_idx)
            count += 1
    
    # Calculate average
    if count > 0:
        avg_val = sum_val / count
    else:
        avg_val = 0.0
    
    # Store result
    if S_0 > 1 and S_1 > 1:
        output_idx = batch_idx * (C * S_0 * S_1) + channel_idx * (S_0 * S_1) + h_out_idx * S_1 + w_out_idx
    elif S_0 > 1:
        output_idx = batch_idx * (C * S_0 * S_1) + channel_idx * (S_0 * S_1) + h_out_idx * S_1
    elif S_1 > 1:
        output_idx = batch_idx * (C * S_0 * S_1) + channel_idx * (S_0 * S_1) + w_out_idx
    else:
        output_idx = channel_idx * (S_0 * S_1)
    
    tl.store(output_ptr + output_idx, avg_val)