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
    # For fractional max pooling, we use a simple approach:
    # We compute the start and end positions in the input
    # based on the output position and kernel size
    
    # Simple mapping: each output element corresponds to a region
    # that is approximately kernel_size in size
    input_start_y = (output_y * input_height) // output_height
    input_start_x = (output_x * input_width) // output_width
    
    # Ensure we don't go out of bounds
    input_end_y = min(input_start_y + kernel_height, input_height)
    input_end_x = min(input_start_x + kernel_width, input_width)
    
    # Find maximum in the region
    max_val = -float('inf')
    max_idx = 0
    
    # Iterate through the region to find max
    for y in range(input_start_y, input_end_y):
        for x in range(input_start_x, input_end_x):
            input_idx = y * input_width + x
            val = tl.load(input_ptr + input_idx)
            if val > max_val:
                max_val = val
                max_idx = input_idx
    
    # Store output and indices
    output_idx = output_y * output_width + output_x
    tl.store(output_ptr + output_idx, max_val)
    if indices_ptr is not None:
        tl.store(indices_ptr + output_idx, max_idx)

def fused_fractional_max_pool2d_with_relu(input, kernel_size, output_size=None, output_ratio=None, return_indices=False):
    # Handle scalar kernel_size
    if isinstance(kernel_size, int):
        kernel_height = kernel_size
        kernel_width = kernel_size
    else:
        kernel_height, kernel_width = kernel_size
    
    # Apply ReLU first
    input_shape = input.shape
    if len(input_shape) == 4:  # batch, channels, height, width
        batch, channels, height, width = input_shape
        input_reshaped = input.view(-1, height, width)
        input_flat = input_reshaped.view(-1, height * width)
    else:
        batch, channels, height, width = 1, 1, input_shape[-2], input_shape[-1]
        input_flat = input.view(-1, height * width)
    
    # Apply ReLU
    relu_out = torch.empty_like(input_flat)
    n = input_flat.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _relu_kernel[grid](input_flat, relu_out, n, BLOCK=block)
    
    # Reshape back to original dimensions
    relu_out = relu_out.view(batch, channels, height, width)
    
    # Determine output size
    if output_size is not None:
        output_height, output_width = output_size
    elif output_ratio is not None:
        ratio_h, ratio_w = output_ratio
        output_height = int(height * ratio_h)
        output_width = int(width * ratio_w)
    else:
        # Default to kernel size
        output_height = math.ceil(height / kernel_height)
        output_width = math.ceil(width / kernel_width)
    
    # Create output tensor
    output = torch.empty(batch, channels, output_height, output_width, device=input.device, dtype=input.dtype)
    
    # Create indices tensor if needed
    indices = None
    if return_indices:
        indices = torch.empty(batch, channels, output_height, output_width, device=input.device, dtype=torch.long)
    
    # Apply fractional max pooling
    if len(input_shape) == 4:
        # Process each batch and channel
        for b in range(batch):
            for c in range(channels):
                # Flatten for kernel processing
                input_flat = relu_out[b, c].view(-1)
                output_flat = output[b, c].view(-1)
                indices_flat = None
                if return_indices:
                    indices_flat = indices[b, c].view(-1)
                
                # Process with Triton kernel
                n = output_flat.numel()
                block = 256
                grid = (1, n)
                _fractional_max_pool2d_kernel[grid](
                    input_flat, output_flat, indices_flat,
                    height, width, output_height, output_width,
                    kernel_height, kernel_width,
                    BLOCK=block
                )
    else:
        # Single batch/channel case
        input_flat = relu_out.view(-1)
        output_flat = output.view(-1)
        indices_flat = None
        if return_indices:
            indices_flat = indices.view(-1)
        
        n = output_flat.numel()
        block = 256
        grid = (1, n)
        _fractional_max_pool2d_kernel[grid](
            input_flat, output_flat, indices_flat,
            height, width, output_height, output_width,
            kernel_height, kernel_width,
            BLOCK=block
        )
    
    # Return appropriate result
    if return_indices:
        return output, indices
    else:
        return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
#     relu_output = F.relu(input)
#     pooled_output = F.fractional_max_pool2d(relu_output, kernel_size=kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices)
#     return pooled_output

def test_fused_fractional_max_pool2d_with_relu():
    results = {}
    
    # Test case 1: Basic functionality with kernel_size and output_size
    input_tensor = torch.randn(1, 1, 8, 8, device='cuda')
    kernel_size = (2, 2)
    output_size = (4, 4)
    results["test_case_1"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size)
    
    # Test case 2: Using output_ratio instead of output_size
    output_ratio = (0.5, 0.5)
    results["test_case_2"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_ratio=output_ratio)
    
    # Test case 3: Return indices along with the pooled output
    results["test_case_3"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size, return_indices=True)
    
    # Test case 4: Larger kernel size
    kernel_size = (3, 3)
    results["test_case_4"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size)
    
    return results

test_results = test_fused_fractional_max_pool2d_with_relu()
