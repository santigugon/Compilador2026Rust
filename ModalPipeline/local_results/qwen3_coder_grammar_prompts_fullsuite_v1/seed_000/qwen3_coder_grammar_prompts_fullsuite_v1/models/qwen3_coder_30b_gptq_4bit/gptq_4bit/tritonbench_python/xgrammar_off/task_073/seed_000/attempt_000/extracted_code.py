import torch
import triton
import triton.language as tl
import math

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    iH, iW, oH, oW, in_channels, out_channels, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output dimensions
    output_size = oH * oW * out_channels
    
    # Each program processes one output element
    if pid >= output_size:
        return
    
    # Calculate which output element this program handles
    out_idx = pid
    out_c = out_idx % out_channels
    out_w = (out_idx // out_channels) % oW
    out_h = (out_idx // (out_channels * oW)) % oH
    
    # Calculate input indices
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Calculate group information
    group_size = in_channels // groups
    group_idx = out_c // (out_channels // groups)
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over kernel dimensions
    for kh in range(kH):
        for kw in range(kW):
            # Calculate input indices
            in_h = in_h_start + kh * dilation_h
            in_w = in_w_start + kw * dilation_w
            
            # Check bounds
            if in_h >= 0 and in_h < iH and in_w >= 0 and in_w < iW:
                # Calculate input channel index within group
                in_c = (out_c % (out_channels // groups)) * group_size
                
                # Load input and weight
                input_val = tl.load(input_ptr + 
                    (group_idx * in_channels + in_c) * iH * iW + 
                    in_h * iW + in_w)
                weight_val = tl.load(weight_ptr + 
                    out_c * kH * kW + 
                    kh * kW + kw)
                acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_c)
        acc += bias_val
    
    # Store result
    tl.store(output_ptr + out_idx, acc)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Handle scalar inputs
    if not isinstance(stride, (tuple, list)):
        stride = (stride, stride)
    if not isinstance(padding, (tuple, list)):
        padding = (padding, padding)
    if not isinstance(dilation, (tuple, list)):
        dilation = (dilation, dilation)
    
    # Get dimensions
    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    
    # Handle padding string
    if isinstance(padding, str):
        if padding == 'valid':
            pad_h, pad_w = 0, 0
        elif padding == 'same':
            # For 'same' padding, we need to calculate padding to make output size match input size
            # This is a simplified version - in practice, this would be more complex
            pad_h, pad_w = 0, 0
    
    # Calculate output height and width
    oH = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(minibatch, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Handle groups
    if groups > 1:
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
    
    # Calculate grid size
    total_elements = minibatch * out_channels * oH * oW
    BLOCK_SIZE = 256
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    if bias is not None:
        _conv2d_kernel[grid](
            input, weight, bias, output,
            iH, iW, oH, oW, in_channels, out_channels, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
            groups, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
        )
    else:
        # Create a dummy bias tensor for kernel compatibility
        dummy_bias = torch.empty(out_channels, device=input.device, dtype=input.dtype)
        _conv2d_kernel[grid](
            input, weight, dummy_bias, output,
            iH, iW, oH, oW, in_channels, out_channels, kH, kW,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
            groups, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
        )
    
    return output
