import torch
import triton
import triton.language as tl
from typing import Union, Tuple, Optional

@triton.jit
def sigmoid_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_height, input_width,
    weight_height, weight_width,
    output_height, output_width,
    in_channels, out_channels,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate output dimensions
    out_h = out_h_idx * BLOCK_SIZE_H
    out_w = out_w_idx * BLOCK_SIZE_W
    
    # Shared memory for input tile
    input_tile = tl.shared.load(input_ptr + batch_idx * in_channels * input_height * input_width + 
                               tl.arange(0, BLOCK_SIZE_H)[:, None] * input_width + 
                               tl.arange(0, BLOCK_SIZE_W)[None, :])
    
    # Initialize accumulator
    acc = 0.0
    
    # Perform convolution
    for k_h in range(weight_height):
        for k_w in range(weight_width):
            # Calculate input indices with padding and dilation
            in_h = out_h * stride_h + k_h * dilation_h - pad_h
            in_w = out_w * stride_w + k_w * dilation_w - pad_w
            
            # Check bounds
            if in_h >= 0 and in_h < input_height and in_w >= 0 and in_w < input_width:
                # Load input and weight
                input_val = input_tile[in_h, in_w]
                weight_val = weight_ptr[out_ch_idx * in_channels * weight_height * weight_width + 
                                       k_h * weight_width + k_w]
                acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        acc += bias_ptr[out_ch_idx]
    
    # Apply sigmoid
    output_val = tl.sigmoid(acc)
    
    # Store result
    output_ptr[batch_idx * out_channels * output_height * output_width + 
              out_ch_idx * output_height * output_width + 
              out_h_idx * output_width + out_w_idx] = output_val

def sigmoid_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Parse stride
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    # Parse padding
    if isinstance(padding, str):
        if padding == 'valid':
            pad_h = pad_w = 0
        elif padding == 'same':
            pad_h = pad_w = 0
        else:
            raise ValueError("Invalid padding string")
    elif isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    # Parse dilation
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    
    # Get dimensions
    batch_size, in_channels, input_height, input_width = input.shape
    out_channels, _, weight_height, weight_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height + 2 * pad_h - (dilation_h * (weight_height - 1) + 1)) // stride_h + 1
    output_width = (input_width + 2 * pad_w - (dilation_w * (weight_width - 1) + 1)) // stride_w + 1
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(batch_size, out_channels, output_height, output_width, device=input.device, dtype=input.dtype)
    
    # Configure grid and block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    
    # Launch kernel
    grid = (
        batch_size,
        out_channels,
        (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    # Launch kernel
    sigmoid_conv2d_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else None,
        output_ptr=out.data_ptr(),
        input_height=input_height,
        input_width=input_width,
        weight_height=weight_height,
        weight_width=weight_width,
        output_height=output_height,
        output_width=output_width,
        in_channels=in_channels,
        out_channels=out_channels,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=groups,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return out
