import torch
import triton
import triton.language as tl
import math

@triton.jit
def _relu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _fractional_max_pool2d_kernel(
    input_ptr, output_ptr, indices_ptr,
    input_height, input_width, output_height, output_width,
    kernel_height, kernel_width,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    
    # Each thread handles one output element
    if tid >= output_height * output_width:
        return
    
    # Calculate output indices
    out_h = tid // output_width
    out_w = tid % output_width
    
    # Calculate input region boundaries
    # For fractional max pooling, we use a simple approach:
    # Map output position to input region using fractional scaling
    input_h_start = (out_h * input_height) // output_height
    input_h_end = ((out_h + 1) * input_height + output_height - 1) // output_height
    input_w_start = (out_w * input_width) // output_width
    input_w_end = ((out_w + 1) * input_width + output_width - 1) // output_width
    
    # Ensure we don't go out of bounds
    input_h_start = tl.minimum(input_h_start, input_height - 1)
    input_h_end = tl.minimum(input_h_end, input_height)
    input_w_start = tl.minimum(input_w_start, input_width - 1)
    input_w_end = tl.minimum(input_w_end, input_width)
    
    # Find maximum value in the region
    max_val = -float('inf')
    max_idx = 0
    
    # Iterate through the region
    for h in range(input_h_start, input_h_end):
        for w in range(input_w_start, input_w_end):
            # Calculate input index
            input_idx = h * input_width + w
            val = tl.load(input_ptr + input_idx)
            if val > max_val:
                max_val = val
                max_idx = input_idx
    
    # Store result
    output_idx = out_h * output_width + out_w
    tl.store(output_ptr + output_idx, max_val)
    
    # Store indices if needed
    if indices_ptr is not None:
        tl.store(indices_ptr + output_idx, max_idx)

def fused_fractional_max_pool2d_with_relu(input, kernel_size, output_size=None, output_ratio=None, return_indices=False):
    # Handle scalar kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    # Handle output_size and output_ratio
    if output_size is not None:
        output_height, output_width = output_size
    elif output_ratio is not None:
        output_height = int(input.shape[2] * output_ratio[0])
        output_width = int(input.shape[3] * output_ratio[1])
    else:
        # Default to kernel_size as output size
        output_height = input.shape[2] // kernel_size[0]
        output_width = input.shape[3] // kernel_size[1]
    
    # Apply ReLU
    input = input.clone()
    input = torch.relu(input)
    
    # Create output tensor
    out = torch.empty(input.shape[0], input.shape[1], output_height, output_width, device=input.device, dtype=input.dtype)
    
    # Create indices tensor if needed
    indices = None
    if return_indices:
        indices = torch.empty(input.shape[0], input.shape[1], output_height, output_width, device=input.device, dtype=torch.long)
    
    # Get dimensions
    batch_size = input.shape[0]
    channels = input.shape[1]
    input_height = input.shape[2]
    input_width = input.shape[3]
    
    # Launch kernel for each batch and channel
    for b in range(batch_size):
        for c in range(channels):
            # Flatten the input for the kernel
            input_flat = input[b, c].flatten()
            output_flat = out[b, c].flatten()
            indices_flat = None
            if return_indices:
                indices_flat = indices[b, c].flatten()
            
            # Launch kernel
            grid = (1, output_height * output_width)
            _fractional_max_pool2d_kernel[grid](
                input_flat, output_flat, indices_flat,
                input_height, input_width, output_height, output_width,
                kernel_size[0], kernel_size[1],
                BLOCK=1024
            )
    
    # Return appropriate result
    if return_indices:
        return out, indices
    else:
        return out
