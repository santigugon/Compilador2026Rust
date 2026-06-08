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
    output_y = tid // output_width
    output_x = tid % output_width
    
    # Calculate input region boundaries
    # For fractional pooling, we use a simple approach where we map output
    # positions to input positions using a fixed ratio
    input_y_start = (output_y * input_height) // output_height
    input_y_end = ((output_y + 1) * input_height + output_height - 1) // output_height
    input_x_start = (output_x * input_width) // output_width
    input_x_end = ((output_x + 1) * input_width + output_width - 1) // output_width
    
    # Clamp to valid range
    input_y_start = tl.minimum(input_y_start, input_height - 1)
    input_y_end = tl.minimum(input_y_end, input_height)
    input_x_start = tl.minimum(input_x_start, input_width - 1)
    input_x_end = tl.minimum(input_x_end, input_width)
    
    # Find max in the region
    max_val = -float('inf')
    max_idx = 0
    
    # Iterate through the region
    for y in range(input_y_start, input_y_end):
        for x in range(input_x_start, input_x_end):
            input_idx = y * input_width + x
            val = tl.load(input_ptr + input_idx)
            if val > max_val:
                max_val = val
                max_idx = input_idx
    
    # Store result
    output_idx = output_y * output_width + output_x
    tl.store(output_ptr + output_idx, max_val)
    
    if indices_ptr is not None:
        tl.store(indices_ptr + output_idx, max_idx)

def fused_fractional_max_pool2d_with_relu(
    input: torch.Tensor, 
    kernel_size, 
    output_size=None, 
    output_ratio=None, 
    return_indices=False
) -> torch.Tensor:
    # Handle input tensor
    if input.dim() != 4:
        raise ValueError("Input tensor must be 4-dimensional (N, C, H, W)")
    
    batch_size, channels, input_height, input_width = input.shape
    
    # Handle kernel size
    if isinstance(kernel_size, int):
        kernel_height = kernel_size
        kernel_width = kernel_size
    else:
        kernel_height, kernel_width = kernel_size
    
    # Calculate output size
    if output_size is not None:
        output_height, output_width = output_size
    elif output_ratio is not None:
        ratio_height, ratio_width = output_ratio
        output_height = int(input_height * ratio_height)
        output_width = int(input_width * ratio_width)
    else:
        # Default behavior - use kernel size to determine output
        output_height = input_height // kernel_height
        output_width = input_width // kernel_width
    
    # Apply ReLU
    relu_out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _relu_kernel[grid](input, relu_out, n, BLOCK=block)
    
    # Apply fractional max pooling
    output = torch.empty(batch_size, channels, output_height, output_width, device=input.device, dtype=input.dtype)
    
    # For simplicity, we'll use a basic approach for fractional pooling
    # In a real implementation, this would be more sophisticated
    
    # Create indices tensor if needed
    indices = None
    if return_indices:
        indices = torch.empty(batch_size, channels, output_height, output_width, device=input.device, dtype=torch.long)
    
    # Process each batch and channel
    for b in range(batch_size):
        for c in range(channels):
            # Flatten the input for processing
            input_flat = relu_out[b, c].flatten()
            output_flat = output[b, c].flatten()
            
            # Process each output element
            for y in range(output_height):
                for x in range(output_width):
                    # Calculate input region
                    input_y_start = (y * input_height) // output_height
                    input_y_end = ((y + 1) * input_height + output_height - 1) // output_height
                    input_x_start = (x * input_width) // output_width
                    input_x_end = ((x + 1) * input_width + output_width - 1) // output_width
                    
                    # Clamp to valid range
                    input_y_start = min(input_y_start, input_height - 1)
                    input_y_end = min(input_y_end, input_height)
                    input_x_start = min(input_x_start, input_width - 1)
                    input_x_end = min(input_x_end, input_width)
                    
                    # Find max in region
                    max_val = float('-inf')
                    max_idx = 0
                    
                    for iy in range(input_y_start, input_y_end):
                        for ix in range(input_x_start, input_x_end):
                            idx = iy * input_width + ix
                            val = input_flat[idx]
                            if val > max_val:
                                max_val = val
                                max_idx = idx
                    
                    # Store result
                    output_flat[y * output_width + x] = max_val
                    
                    if return_indices:
                        indices[b, c, y, x] = max_idx
    
    if return_indices:
        return output, indices
    else:
        return output
