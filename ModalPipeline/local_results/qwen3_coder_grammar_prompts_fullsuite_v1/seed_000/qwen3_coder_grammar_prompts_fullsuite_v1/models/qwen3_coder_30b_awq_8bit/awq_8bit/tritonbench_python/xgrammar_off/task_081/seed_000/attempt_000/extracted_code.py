import torch
import triton
import triton.language as tl
from typing import Union, Tuple

@triton.jit
def _sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    input_batch,
    input_height,
    input_width,
    output_height,
    output_width,
    input_batch_stride,
    input_height_stride,
    input_width_stride,
    output_batch_stride,
    output_height_stride,
    output_width_stride,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    batch_id = tl.program_id(0)
    h_id = tl.program_id(1)
    w_id = tl.program_id(2)
    
    # Calculate output indices
    output_h = h_id * BLOCK_H + tl.arange(0, BLOCK_H)
    output_w = w_id * BLOCK_W + tl.arange(0, BLOCK_W)
    
    # Create masks for valid output indices
    h_mask = output_h < output_height
    w_mask = output_w < output_width
    mask = h_mask[:, None] & w_mask[None, :]
    
    # Calculate input region boundaries
    h_start = (output_h * input_height) // output_height
    h_end = ((output_h + 1) * input_height + output_height - 1) // output_height
    w_start = (output_w * input_width) // output_width
    w_end = ((output_w + 1) * input_width + output_width - 1) // output_width
    
    # Calculate average pooling
    avg_sum = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    count = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    for h in range(BLOCK_H):
        for w in range(BLOCK_W):
            if h_mask[h] and w_mask[w]:
                h_s = h_start[h]
                h_e = h_end[h]
                w_s = w_start[w]
                w_e = w_end[w]
                region_count = (h_e - h_s) * (w_e - w_s)
                if region_count > 0:
                    # Compute sum over the region
                    region_sum = 0.0
                    for ih in range(h_s, h_e):
                        for iw in range(w_s, w_e):
                            input_idx = batch_id * input_batch_stride + ih * input_height_stride + iw * input_width_stride
                            region_sum += tl.load(input_ptr + input_idx, mask=True)
                    avg_sum[h, w] = region_sum
                    count[h, w] = region_count
    
    # Compute average
    avg = avg_sum / count
    
    # Apply sigmoid
    sigmoid_result = 1.0 / (1.0 + tl.exp(-avg))
    
    # Store result
    output_idx = batch_id * output_batch_stride + h_id * BLOCK_H * output_height_stride + w_id * BLOCK_W * output_width_stride
    tl.store(output_ptr + output_idx, sigmoid_result, mask=mask)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    # Handle output_size
    if isinstance(output_size, int):
        output_height = output_size
        output_width = output_size
    else:
        output_height, output_width = output_size
    
    # Get input dimensions
    batch_size, channels, input_height, input_width = input.shape
    
    # Create output tensor
    output = torch.empty(batch_size, channels, output_height, output_width, dtype=input.dtype, device=input.device)
    
    # Define block size
    BLOCK_H = 16
    BLOCK_W = 16
    
    # Launch kernel
    grid = (
        batch_size,
        triton.cdiv(output_height, BLOCK_H),
        triton.cdiv(output_width, BLOCK_W)
    )
    
    # Get strides
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride = input.stride()
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride = output.stride()
    
    _sigmoid_adaptive_avg_pool2d_kernel[grid](
        input,
        output,
        batch_size,
        input_height,
        input_width,
        output_height,
        output_width,
        input_batch_stride,
        input_height_stride,
        input_width_stride,
        output_batch_stride,
        output_height_stride,
        output_width_stride,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    return output
