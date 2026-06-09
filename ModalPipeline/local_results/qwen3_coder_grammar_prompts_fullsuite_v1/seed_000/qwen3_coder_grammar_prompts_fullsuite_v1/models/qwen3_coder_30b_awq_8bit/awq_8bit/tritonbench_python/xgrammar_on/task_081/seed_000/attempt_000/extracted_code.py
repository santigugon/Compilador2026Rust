import torch
import triton
import triton.language as tl
from typing import Union, Tuple

@triton.jit
def _adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    input_height, input_width,
    output_height, output_width,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate the start and end indices for the input region
    h_start = (pid_h * input_height) // output_height
    h_end = ((pid_h + 1) * input_height + output_height - 1) // output_height
    w_start = (pid_w * input_width) // output_width
    w_end = ((pid_w + 1) * input_width + output_width - 1) // output_width
    
    # Calculate the number of elements in the region
    region_size = (h_end - h_start) * (w_end - w_start)
    
    # Load input values in the region
    sum_val = 0.0
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            input_idx = h * input_stride_0 + w * input_stride_1
            sum_val += tl.load(input_ptr + input_idx)
    
    # Calculate average
    avg_val = sum_val / region_size
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-avg_val))
    
    # Store result
    output_idx = pid_h * output_stride_0 + pid_w * output_stride_1
    tl.store(output_ptr + output_idx, sigmoid_val)

@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    # Handle output_size
    if isinstance(output_size, int):
        output_height = output_size
        output_width = output_size
    else:
        output_height, output_width = output_size
    
    # Get input dimensions
    input_height, input_width = input.shape[-2], input.shape[-1]
    
    # Create output tensor
    if len(input.shape) == 4:
        batch_size = input.shape[0]
        out = torch.empty((batch_size, input.shape[1], output_height, output_width), dtype=input.dtype, device=input.device)
        # Process each batch
        for b in range(batch_size):
            for c in range(input.shape[1]):
                # Flatten the input and output for kernel processing
                input_flat = input[b, c].contiguous()
                out_flat = out[b, c].contiguous()
                
                # Launch kernel
                grid_h = triton.cdiv(output_height, 16)
                grid_w = triton.cdiv(output_width, 16)
                grid = (grid_h, grid_w)
                
                _adaptive_avg_pool2d_kernel[grid](
                    input_flat, out_flat,
                    input_height, input_width,
                    output_height, output_width,
                    input_flat.stride(0), input_flat.stride(1),
                    out_flat.stride(0), out_flat.stride(1),
                    BLOCK_H=16,
                    BLOCK_W=16
                )
    else:
        # For 3D input (no batch dimension)
        out = torch.empty((input.shape[0], output_height, output_width), dtype=input.dtype, device=input.device)
        
        # Flatten the input and output for kernel processing
        input_flat = input.contiguous()
        out_flat = out.contiguous()
        
        # Launch kernel
        grid_h = triton.cdiv(output_height, 16)
        grid_w = triton.cdiv(output_width, 16)
        grid = (grid_h, grid_w)
        
        _adaptive_avg_pool2d_kernel[grid](
            input_flat, out_flat,
            input_height, input_width,
            output_height, output_width,
            input_flat.stride(0), input_flat.stride(1),
            out_flat.stride(0), out_flat.stride(1),
            BLOCK_H=16,
            BLOCK_W=16
        )
    
    # Apply sigmoid to the final output
    out = out.view(-1)
    out_sigmoid = torch.empty_like(out)
    n = out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _sigmoid_kernel[grid](out, out_sigmoid, n, BLOCK=block)
    return out_sigmoid.view(out.shape)