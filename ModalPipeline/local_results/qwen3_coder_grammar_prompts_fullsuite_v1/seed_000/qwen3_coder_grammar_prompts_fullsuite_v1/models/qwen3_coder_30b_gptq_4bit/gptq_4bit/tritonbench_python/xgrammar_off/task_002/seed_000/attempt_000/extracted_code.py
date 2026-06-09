import torch
import triton
import triton.language as tl
import math

@triton.jit
def _conv2d_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    iH, iW, oH, oW, in_channels, out_channels, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    output_size = oH * oW * out_channels
    
    # Each program handles one output element
    if pid >= output_size:
        return
    
    # Calculate which output element this program handles
    out_c = pid % out_channels
    out_w = (pid // out_channels) % oW
    out_h = (pid // out_channels) // oW
    
    # Calculate input region
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Initialize accumulator
    output_val = 0.0
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific indices
        group_in_channels = in_channels // groups
        group_out_channels = out_channels // groups
        
        # Calculate group-specific output channel
        group_out_c = out_c % group_out_channels
        
        # Loop over kernel elements
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input coordinates
                in_h = in_h_start + kh * dilation_h
                in_w = in_w_start + kw * dilation_w
                
                # Check bounds
                if in_h >= 0 and in_h < iH and in_w >= 0 and in_w < iW:
                    # Calculate input channel
                    in_c = g * group_in_channels + (kh * kW + kw) % group_in_channels
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + 
                                       (g * group_in_channels + in_c) * (iH * iW) + 
                                       in_h * iW + in_w)
                    weight_val = tl.load(weight_ptr + 
                                        (g * group_out_channels + group_out_c) * (kH * kW) + 
                                        kh * kW + kw)
                    output_val += input_val * weight_val
                else:
                    # Padding case - treat as 0
                    pass
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c)
        output_val += bias_val
    
    # Apply sigmoid
    output_val = 1.0 / (1.0 + tl.exp(-output_val))
    
    # Store result
    tl.store(output_ptr + pid, output_val)

def _calculate_output_size(iH, iW, kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w):
    oH = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    return oH, oW

def _parse_padding(padding, iH, iW):
    if isinstance(padding, str):
        if padding == 'valid':
            return 0, 0
        elif padding == 'same':
            return (iH + 1) // 2, (iW + 1) // 2
        else:
            raise ValueError("Invalid padding string")
    elif isinstance(padding, int):
        return padding, padding
    elif isinstance(padding, tuple):
        if len(padding) == 2:
            return padding[0], padding[1]
        else:
            raise ValueError("Invalid padding tuple")
    else:
        raise ValueError("Invalid padding type")

def _parse_stride(stride):
    if isinstance(stride, int):
        return stride, stride
    elif isinstance(stride, tuple):
        if len(stride) == 2:
            return stride[0], stride[1]
        else:
            raise ValueError("Invalid stride tuple")
    else:
        raise ValueError("Invalid stride type")

def _parse_dilation(dilation):
    if isinstance(dilation, int):
        return dilation, dilation
    elif isinstance(dilation, tuple):
        if len(dilation) == 2:
            return dilation[0], dilation[1]
        else:
            raise ValueError("Invalid dilation tuple")
    else:
        raise ValueError("Invalid dilation type")

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Parse inputs
    iH, iW = input.shape[2], input.shape[3]
    kH, kW = weight.shape[2], weight.shape[3]
    out_channels = weight.shape[0]
    in_channels = weight.shape[1] * groups
    
    # Parse stride
    stride_h, stride_w = _parse_stride(stride)
    
    # Parse padding
    pad_h, pad_w = _parse_padding(padding, iH, iW)
    
    # Parse dilation
    dilation_h, dilation_w = _parse_dilation(dilation)
    
    # Calculate output size
    oH, oW = _calculate_output_size(iH, iW, kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w)
    
    # Create output tensor
    if out is not None:
        output = out
    else:
        output = torch.empty(input.shape[0], out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Handle bias
    if bias is not None:
        bias = bias.to(input.device)
    
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Prepare kernel launch parameters
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 16
    
    # Calculate grid size
    total_elements = output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3]
    grid_size = total_elements
    
    # Launch kernel
    if grid_size > 0:
        _conv2d_sigmoid_kernel[grid_size](
            input, weight, bias, output,
            iH, iW, oH, oW, in_channels, out_channels, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
            groups, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
        )
    
    return output
