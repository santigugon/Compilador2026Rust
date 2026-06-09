import torch
import triton
import triton.language as tl

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Handle scalar inputs
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    
    # Get input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    padH, padW = padding
    strideH, strideW = stride
    dilationH, dilationW = dilation
    
    # Apply padding
    if padH > 0 or padW > 0:
        input = torch.nn.functional.pad(input, (padW, padW, padH, padH))
        iH, iW = input.shape[2], input.shape[3]
    
    # Calculate output height and width
    oH = (iH - (kH - 1) * dilationH - 1) // strideH + 1
    oW = (iW - (kW - 1) * dilationW - 1) // strideW + 1
    
    # Reshape input for grouped convolution
    if groups > 1:
        input = input.view(batch_size, groups, in_channels // groups, iH, iW)
        weight = weight.view(groups, out_channels // groups, in_channels // groups, kH, kW)
        
    # Initialize output tensor
    output = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Launch kernel
    if groups == 1:
        _conv2d_kernel[1, 1](input, weight, output, 
                            iH, iW, oH, oW, kH, kW,
                            strideH, strideW, dilationH, dilationW,
                            BLOCK_SIZE=32)
    else:
        # For grouped convolution, we need to handle each group separately
        for g in range(groups):
            input_group = input[:, g, :, :, :]
            weight_group = weight[g, :, :, :, :]
            output_group = output[:, g * (out_channels // groups):(g + 1) * (out_channels // groups), :, :]
            _conv2d_kernel[1, 1](input_group, weight_group, output_group,
                                iH, iW, oH, oW, kH, kW,
                                strideH, strideW, dilationH, dilationW,
                                BLOCK_SIZE=32)
    
    # Add bias if provided
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    
    return output

@triton.jit
def _conv2d_kernel(input_ptr, weight_ptr, output_ptr,
                   iH: tl.constexpr, iW: tl.constexpr,
                   oH: tl.constexpr, oW: tl.constexpr,
                   kH: tl.constexpr, kW: tl.constexpr,
                   strideH: tl.constexpr, strideW: tl.constexpr,
                   dilationH: tl.constexpr, dilationW: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_h_idx = tl.program_id(1)
    out_w_idx = tl.program_id(2)
    
    # Shared memory for input tile
    input_tile = tl.shared_ptr(input_ptr + batch_idx * iH * iW, (iH, iW), dtype=tl.float32)
    
    # Load weight
    weight = tl.load(weight_ptr)
    
    # Compute output value
    out_val = 0.0
    for kh in range(kH):
        for kw in range(kW):
            # Calculate input indices
            ih = out_h_idx * strideH + kh * dilationH
            iw = out_w_idx * strideW + kw * dilationW
            
            # Check bounds
            if ih < iH and iw < iW:
                input_val = tl.load(input_ptr + batch_idx * iH * iW + ih * iW + iw)
                weight_val = tl.load(weight_ptr + kh * kW + kw)
                out_val += input_val * weight_val
    
    # Store result
    tl.store(output_ptr + batch_idx * oH * oW + out_h_idx * oW + out_w_idx, out_val)