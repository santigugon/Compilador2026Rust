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
    BLOCK_H: tl.constexpr, 
    BLOCK_W: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    if pid_h >= output_height or pid_w >= output_width:
        return
    
    # Calculate the start position in the input tensor
    start_h = pid_h * kernel_height
    start_w = pid_w * kernel_width
    
    # Initialize max value and index
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Iterate through the kernel window
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            h = start_h + kh
            w = start_w + kw
            
            # Load input value
            input_idx = h * input_width + w
            val = tl.load(input_ptr + input_idx, mask=(h < input_height) & (w < input_width), other=-float('inf'))
            
            # Update max if current value is greater
            mask = val > max_val
            max_val = tl.where(mask, val, max_val)
            max_idx = tl.where(mask, input_idx, max_idx)
    
    # Store output and indices
    output_idx = pid_h * output_width + pid_w
    tl.store(output_ptr + output_idx, max_val)
    if indices_ptr is not None:
        tl.store(indices_ptr + output_idx, max_idx)

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    # Handle input tensor
    if input.dim() != 4:
        raise ValueError("Input tensor must be 4-dimensional (N, C, H, W)")
    
    N, C, H, W = input.shape
    
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
        ratio_h, ratio_w = output_ratio
        output_height = int(H * ratio_h)
        output_width = int(W * ratio_w)
    else:
        # Default to fractional max pooling with kernel size
        output_height = H // kernel_height
        output_width = W // kernel_width
    
    # Apply ReLU
    relu_out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _relu_kernel[grid](input, relu_out, n, BLOCK=block)
    
    # Prepare output tensors
    output = torch.empty(N, C, output_height, output_width, device=input.device, dtype=input.dtype)
    
    if return_indices:
        indices = torch.empty(N, C, output_height, output_width, device=input.device, dtype=torch.int32)
    else:
        indices = None
    
    # Apply fractional max pooling
    if output_height > 0 and output_width > 0:
        # Create a grid for the 2D pooling operation
        grid_h = triton.cdiv(output_height, 16)
        grid_w = triton.cdiv(output_width, 16)
        grid = (grid_h, grid_w)
        
        # Launch kernel
        _fractional_max_pool2d_kernel[grid](
            relu_out,
            output,
            indices,
            H,
            W,
            output_height,
            output_width,
            kernel_height,
            kernel_width,
            BLOCK_H=16,
            BLOCK_W=16
        )
    
    if return_indices:
        return output, indices
    else:
        return output
