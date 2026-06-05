import torch
import triton
import triton.language as tl

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
    input_ptr, 
    output_ptr, 
    indices_ptr,
    input_height, 
    input_width, 
    output_height, 
    output_width,
    kernel_height, 
    kernel_width,
    BLOCK_HEIGHT: tl.constexpr, 
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    if pid_h >= output_height or pid_w >= output_width:
        return
    
    # Calculate the start position in the input tensor
    start_h = pid_h * kernel_height
    start_w = pid_w * kernel_width
    
    # Load the kernel_size for this block
    max_val = -float('inf')
    max_idx = 0
    
    # Loop through the kernel window to find max
    for h in range(kernel_height):
        for w in range(kernel_width):
            input_h = start_h + h
            input_w = start_w + w
            if input_h < input_height and input_w < input_width:
                input_offset = input_h * input_width + input_w
                val = tl.load(input_ptr + input_offset)
                if val > max_val:
                    max_val = val
                    max_idx = input_offset
    
    # Store the result
    output_offset = pid_h * output_width + pid_w
    tl.store(output_ptr + output_offset, max_val)
    
    if indices_ptr is not None:
        tl.store(indices_ptr + output_offset, max_idx)

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False):
    # Handle kernel_size
    if isinstance(kernel_size, int):
        kernel_height = kernel_size
        kernel_width = kernel_size
    else:
        kernel_height, kernel_width = kernel_size
    
    # Apply ReLU first
    input_re = torch.relu(input)
    
    # Get input dimensions
    input_height, input_width = input_re.shape[-2], input_re.shape[-1]
    
    # Calculate output size if not provided
    if output_size is not None:
        output_height, output_width = output_size
    elif output_ratio is not None:
        output_height = int(input_height * output_ratio[0])
        output_width = int(input_width * output_ratio[1])
    else:
        # Default to kernel size
        output_height = input_height // kernel_height
        output_width = input_width // kernel_width
    
    # Create output tensor
    out = torch.empty(input_re.shape[:-2] + (output_height, output_width), device=input.device, dtype=input.dtype)
    
    # Create indices tensor if needed
    indices = None
    if return_indices:
        indices = torch.empty(input_re.shape[:-2] + (output_height, output_width), device=input.device, dtype=torch.long)
    
    # Handle batch dimensions
    batch_size = input_re.shape[:-2]
    total_batch_elements = 1
    for dim in batch_size:
        total_batch_elements *= dim
    
    # Flatten batch dimensions for processing
    if len(batch_size) > 0:
        input_flat = input_re.view(-1, input_height, input_width)
        out_flat = out.view(-1, output_height, output_width)
        if return_indices:
            indices_flat = indices.view(-1, output_height, output_width)
    else:
        input_flat = input_re.unsqueeze(0)
        out_flat = out.unsqueeze(0)
        if return_indices:
            indices_flat = indices.unsqueeze(0)
    
    # Process each batch element
    for i in range(total_batch_elements):
        # Get the current batch element
        if len(batch_size) > 0:
            input_batch = input_flat[i]
            out_batch = out_flat[i]
            if return_indices:
                indices_batch = indices_flat[i]
        else:
            input_batch = input_flat[0]
            out_batch = out_flat[0]
            if return_indices:
                indices_batch = indices_flat[0]
        
        # Launch kernel
        grid_h = triton.cdiv(output_height, 16)
        grid_w = triton.cdiv(output_width, 16)
        grid = (grid_h, grid_w)
        
        # For simplicity, we'll use a single block size for now
        BLOCK_SIZE = 256
        _fractional_max_pool2d_kernel[grid](
            input_batch, 
            out_batch, 
            indices_batch if return_indices else None,
            input_height, 
            input_width, 
            output_height, 
            output_width,
            kernel_height, 
            kernel_width,
            BLOCK_HEIGHT=16, 
            BLOCK_WIDTH=16,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    # Reshape back to original batch dimensions
    if len(batch_size) > 0:
        out = out.view(input.shape[:-2] + (output_height, output_width,))
        if return_indices:
            indices = indices.view(input.shape[:-2] + (output_height, output_width,))
    
    if return_indices:
        return out, indices
    else:
        return out
