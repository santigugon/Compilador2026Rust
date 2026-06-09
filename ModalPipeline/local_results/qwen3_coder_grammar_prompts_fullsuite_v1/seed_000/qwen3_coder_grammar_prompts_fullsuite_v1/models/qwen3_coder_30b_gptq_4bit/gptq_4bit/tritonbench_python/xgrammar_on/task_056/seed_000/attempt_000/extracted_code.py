import torch
import triton
import triton.language as tl

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    # Handle scalar kernel_size
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    # Determine output size
    if output_size is not None:
        output_h, output_w = output_size
    elif output_ratio is not None:
        output_h = int(input.shape[2] * output_ratio[0])
        output_w = int(input.shape[3] * output_ratio[1])
    else:
        # Default to 1x1 output
        output_h, output_w = 1, 1
    
    # Apply ReLU
    input_relu = torch.relu(input)
    
    # Initialize output tensor
    out = torch.empty(input.shape[0], input.shape[1], output_h, output_w, device=input.device, dtype=input.dtype)
    
    # Initialize indices tensor if needed
    if return_indices:
        indices = torch.empty(input.shape[0], input.shape[1], output_h, output_w, device=input.device, dtype=torch.long)
    else:
        indices = None
    
    # Get input dimensions
    batch, channels, input_h, input_w = input.shape
    
    # Calculate block size
    BLOCK_H = 16
    BLOCK_W = 16
    
    # Launch kernel
    grid = (triton.cdiv(output_h, BLOCK_H), triton.cdiv(output_w, BLOCK_W), batch * channels)
    
    _fused_fractional_max_pool2d_with_relu_kernel[grid](
        input_relu, out, indices,
        input_h, input_w, output_h, output_w,
        kernel_size[0], kernel_size[1],
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_H, BLOCK_W
    )
    
    if return_indices:
        return out, indices
    else:
        return out

@triton.jit
def _fused_fractional_max_pool2d_with_relu_kernel(
    input_ptr, output_ptr, indices_ptr,
    input_h, input_w, output_h, output_w,
    kernel_h, kernel_w,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Calculate output indices
    output_h_idx = pid * BLOCK_H
    output_w_idx = pid_w * BLOCK_W
    
    # Calculate input indices
    input_h_start = output_h_idx * kernel_h
    input_w_start = output_w_idx * kernel_w
    
    # Calculate block size
    block_h = min(BLOCK_H, output_h - output_h_idx)
    block_w = min(BLOCK_W, output_w - output_w_idx)
    
    # Calculate output position
    output_offset = pid_c * output_stride_1 + output_h_idx * output_stride_2 + output_w_idx * output_stride_3
    
    # Calculate input position
    input_offset = pid_c * input_stride_1 + input_h_start * input_stride_2 + input_w_start * input_stride_3
    
    # Initialize max value and index
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    max_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Iterate through kernel
    for i in range(kernel_h):
        for j in range(kernel_w):
            # Calculate input position
            input_h_pos = input_h_start + i
            input_w_pos = input_w_start + j
            
            # Check bounds
            if input_h_pos < input_h and input_w_pos < input_w:
                # Load input value
                input_val = tl.load(input_ptr + input_offset + i * input_stride_2 + j * input_stride_3)
                
                # Update max
                mask = input_val > max_val
                max_val = tl.where(mask, input_val, max_val)
                max_idx = tl.where(mask, i * kernel_w + j, max_idx)
            
    # Store output
    tl.store(output_ptr + output_offset, max_val)
    
    # Store indices if needed
    if indices_ptr is not None:
        tl.store(indices_ptr + output_offset, max_idx)