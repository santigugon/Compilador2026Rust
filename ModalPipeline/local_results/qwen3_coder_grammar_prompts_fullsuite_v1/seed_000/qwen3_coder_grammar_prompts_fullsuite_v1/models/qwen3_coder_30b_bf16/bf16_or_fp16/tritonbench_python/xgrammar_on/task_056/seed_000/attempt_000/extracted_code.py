import torch
import triton
import triton.language as tl

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    if input.dim() != 4:
        raise ValueError("Input tensor must be 4-dimensional (N, C, H, W)")
    
    if output_size is not None and output_ratio is not None:
        raise ValueError("Only one of output_size or output_ratio should be specified")
    
    if isinstance(kernel_size, int):
        kernel_h, kernel_w = kernel_size, kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    batch_size, channels, input_h, input_w = input.shape
    
    if output_ratio is not None:
        output_h = int(input_h * output_ratio[0])
        output_w = int(input_w * output_ratio[1])
    elif output_size is not None:
        output_h, output_w = output_size
    else:
        output_h = (input_h + kernel_h - 1) // kernel_h
        output_w = (input_w + kernel_w - 1) // kernel_w
    
    output = torch.empty((batch_size, channels, output_h, output_w), dtype=torch.float32, device=input.device)
    indices = torch.empty((batch_size, channels, output_h, output_w), dtype=torch.int32, device=input.device) if return_indices else None
    
    # Launch kernel
    grid = (batch_size, channels, output_h, output_w)
    _fused_fractional_max_pool2d_with_relu_kernel[grid](
        input, output, indices,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        input_h, input_w, output_h, output_w, kernel_h, kernel_w,
        return_indices
    )
    
    return (output, indices) if return_indices else output

@triton.jit
def _fused_fractional_max_pool2d_with_relu_kernel(
    input_ptr, output_ptr, indices_ptr,
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    input_h, input_w, output_h, output_w, kernel_h, kernel_w,
    return_indices
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)
    
    # Calculate fractional pooling indices
    # This is a simplified version - in practice, you'd want more sophisticated
    # fractional pooling logic that properly handles the fractional aspect
    h_start = h_idx * kernel_h
    w_start = w_idx * kernel_w
    h_end = min(h_start + kernel_h, input_h)
    w_end = min(w_start + kernel_w, input_w)
    
    # Apply ReLU
    max_val = -float('inf')
    max_idx = -1
    
    for ih in range(h_start, h_end):
        for iw in range(w_start, w_end):
            input_offset = batch_idx * input_batch_stride + channel_idx * input_channel_stride + ih * input_h_stride + iw * input_w_stride
            val = tl.load(input_ptr + input_offset)
            val = tl.maximum(val, 0.0)  # ReLU
            if val > max_val:
                max_val = val
                max_idx = ih * input_w + iw
    
    # Store output
    output_offset = batch_idx * output_batch_stride + channel_idx * output_channel_stride + h_idx * output_h_stride + w_idx * output_w_stride
    tl.store(output_ptr + output_offset, max_val)
    
    # Store indices if requested
    if return_indices:
        tl.store(indices_ptr + output_offset, max_idx)