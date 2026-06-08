import torch
import triton
import triton.language as tl

@triton.jit
def _adaptive_avg_pool2d_kernel(
    input_ptr, 
    output_ptr, 
    input_shape, 
    output_shape,
    H_in: tl.constexpr, 
    W_in: tl.constexpr,
    H_out: tl.constexpr, 
    W_out: tl.constexpr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block index
    pid = tl.program_id(0)
    
    # Calculate which output element this block handles
    output_idx = pid * BLOCK_SIZE
    
    # Handle the case where we have multiple output elements per block
    for i in range(BLOCK_SIZE):
        if output_idx + i >= H_out * W_out:
            break
            
        # Calculate which output position (h, w) this block is handling
        out_h = (output_idx + i) // W_out
        out_w = (output_idx + i) % W_out
        
        # Calculate the input region for this output element
        h_start = (out_h * H_in) // H_out
        h_end = ((out_h + 1) * H_in + H_out - 1) // H_out
        w_start = (out_w * W_in) // W_out
        w_end = ((out_w + 1) * W_in + W_out - 1) // W_out
        
        # Calculate the number of elements in this region
        h_size = h_end - h_start
        w_size = w_end - w_start
        
        # Initialize sum
        sum_val = 0.0
        
        # Accumulate values from the input region
        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                # Calculate input index
                input_idx = h * W_in + w
                # Load input value
                input_val = tl.load(input_ptr + input_idx, mask=True)
                sum_val += input_val
        
        # Calculate average
        avg_val = sum_val / (h_size * w_size)
        
        # Store result
        output_idx_full = out_h * W_out + out_w
        tl.store(output_ptr + output_idx_full, avg_val, mask=True)

def adaptive_avg_pool2d(input, output_size):
    # Handle different input shapes
    if input.dim() == 3:
        # (C, H_in, W_in)
        N = 1
        C = input.shape[0]
        H_in = input.shape[1]
        W_in = input.shape[2]
        input_flat = input.view(-1)
    else:
        # (N, C, H_in, W_in)
        N = input.shape[0]
        C = input.shape[1]
        H_in = input.shape[2]
        W_in = input.shape[3]
        input_flat = input.view(-1)
    
    # Handle output size
    if isinstance(output_size, int):
        H_out = output_size
        W_out = output_size
    elif isinstance(output_size, tuple) and len(output_size) == 2:
        H_out = output_size[0]
        W_out = output_size[1]
    else:
        raise ValueError("output_size must be an int or a tuple of two ints")
    
    # Create output tensor
    if N == 1:
        output = torch.empty((C, H_out, W_out), dtype=input.dtype, device=input.device)
    else:
        output = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=input.device)
    
    # Flatten output for kernel
    output_flat = output.view(-1)
    
    # Calculate total elements in output
    total_elements = H_out * W_out
    
    # Launch kernel
    block_size = 256
    grid_size = triton.cdiv(total_elements, block_size)
    
    # Call kernel
    _adaptive_avg_pool2d_kernel[grid_size](
        input_flat,
        output_flat,
        input.shape,
        output.shape,
        H_in,
        W_in,
        H_out,
        W_out,
        C,
        N,
        block_size
    )
    
    return output
