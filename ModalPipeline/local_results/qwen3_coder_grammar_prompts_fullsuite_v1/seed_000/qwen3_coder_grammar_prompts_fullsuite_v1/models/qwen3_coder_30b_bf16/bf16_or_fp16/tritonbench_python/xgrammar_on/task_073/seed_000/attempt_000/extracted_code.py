import torch
import triton
import triton.language as tl

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Handle scalar inputs
    if not isinstance(stride, (tuple, list)):
        stride = (stride, stride)
    if not isinstance(padding, (tuple, list)):
        padding = (padding, padding)
    if not isinstance(dilation, (tuple, list)):
        dilation = (dilation, dilation)
    
    # Get input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    padH, padW = padding
    strideH, strideW = stride
    dilationH, dilationW = dilation
    
    # Apply dilation and padding
    dilated_kH = kH + (kH - 1) * (dilationH - 1)
    dilated_kW = kW + (kW - 1) * (dilationW - 1)
    
    oH = (iH + 2 * padH - dilated_kH) // strideH + 1
    oW = (iW + 2 * padW - dilated_kW) // strideW + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=input.dtype)
    
    # Handle groups
    if groups > 1:
        # Split input and weight into groups
        input_per_group = in_channels // groups
        weight_per_group = out_channels // groups
        
        for g in range(groups):
            input_group = input[:, g*input_per_group:(g+1)*input_per_group, :, :]
            weight_group = weight[g*weight_per_group:(g+1)*weight_per_group, :, :, :]
            
            # Compute convolution for this group
            _conv2d_group_kernel(input_group, weight_group, output[:, g*weight_per_group:(g+1)*weight_per_group, :, :],
                                bias[g*weight_per_group:(g+1)*weight_per_group] if bias is not None else None,
                                stride, padding, dilation, batch_size, in_channels, out_channels, iH, iW, oH, oW, kH, kW)
    else:
        # Single group case
        _conv2d_group_kernel(input, weight, output, bias, stride, padding, dilation, batch_size, in_channels, out_channels, iH, iW, oH, oW, kH, kW)
    
    return output

@triton.jit
def _conv2d_group_kernel(input_ptr, weight_ptr, output_ptr, bias_ptr, stride, padding, dilation, batch_size, in_channels, out_channels, iH, iW, oH, oW, kH, kW):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate output position
    strideH, strideW = stride
    padH, padW = padding
    dilationH, dilationW = dilation
    
    # Calculate output coordinates
    out_h = out_h_idx
    out_w = out_w_idx
    
    # Calculate input coordinates
    in_h_start = out_h * strideH - padH
    in_w_start = out_w * strideW - padW
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Perform convolution
    for kh in range(kH):
        for kw in range(kW):
            # Calculate input position
            in_h = in_h_start + kh * dilationH
            in_w = in_w_start + kw * dilationW
            
            # Check bounds
            if in_h >= 0 and in_h < iH and in_w >= 0 and in_w < iW:
                # Load input and weight
                input_val = tl.load(input_ptr + batch_idx * (in_channels * iH * iW) + 
                                   in_h * iW + in_w)
                weight_val = tl.load(weight_ptr + out_ch_idx * (in_channels * kH * kW) + 
                                    kh * kW + kw)
                acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + out_ch_idx)
    
    # Store result
    output_val = acc.to(tl.float32)
    tl.store(output_ptr + batch_idx * (out_channels * oH * oW) + 
            out_ch_idx * (oH * oW) + out_h * oW + out_w, output_val)