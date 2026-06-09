import torch
import triton
import triton.language as tl

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: int) -> torch.Tensor:
    if not isinstance(output_size, int):
        output_size = output_size[0] if isinstance(output_size, tuple) else output_size
    
    batch, channels, height, width = input.shape
    output_height, output_width = output_size, output_size
    
    # Create output tensor
    output = torch.empty(batch, channels, output_height, output_width, device=input.device, dtype=input.dtype)
    
    # Calculate block size
    BLOCK = 256
    grid = (triton.cdiv(batch * channels * output_height * output_width, BLOCK),)
    
    # Launch kernel
    _sigmoid_adaptive_avg_pool2d_kernel[grid](
        input, output,
        height, width,
        output_height, output_width,
        batch, channels,
        BLOCK
    )
    
    return output

@triton.jit
def _sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    height: tl.constexpr, width: tl.constexpr,
    output_height: tl.constexpr, output_width: tl.constexpr,
    batch: tl.constexpr, channels: tl.constexpr,
    BLOCK: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate total elements
    total_elements = batch * channels * output_height * output_width
    
    # Calculate which element we're processing
    element_id = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Mask for valid elements
    mask = element_id < total_elements
    
    # Calculate indices
    if mask:
        # Flatten the output indices
        flat_idx = element_id
        batch_idx = flat_idx // (channels * output_height * output_width)
        remaining = flat_idx % (channels * output_height * output_width)
        channel_idx = remaining // (output_height * output_width)
        remaining = remaining % (output_height * output_width)
        out_h_idx = remaining // output_width
        out_w_idx = remaining % output_width
        
        # Calculate input region bounds
        h_start = (out_h_idx * height) // output_height
        h_end = ((out_h_idx + 1) * height) // output_height
        w_start = (out_w_idx * width) // output_width
        w_end = ((out_w_idx + 1) * width) // output_width
        
        # Calculate average
        sum_val = 0.0
        count = 0
        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                sum_val += tl.load(input_ptr + batch_idx * (channels * height * width) + channel_idx * (height * width) + h * width + w)
                count += 1
        
        # Avoid division by zero
        if count > 0:
            avg_val = sum_val / count
        else:
            avg_val = 0.0
        
        # Apply sigmoid
        sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
        
        # Store result
        tl.store(output_ptr + flat_idx, sigmoid_val)