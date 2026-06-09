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

    # Get dimensions
    batch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    padH, padW = padding
    strideH, strideW = stride
    dilationH, dilationW = dilation
    
    # Apply dilation to kernel size
    actual_kH = (kH - 1) * dilationH + 1
    actual_kW = (kW -1) * dilationW + 1
    
    oH = (iH + 2 * padH - actual_kH) // strideH + 1
    oW = (iW + 2 * padW - actual_kW) // strideW + 1
    
    # Handle groups
    if groups > 1:
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        
    # Create output tensor
    output = torch.empty(batch, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Handle bias
    if bias is not None:
        assert bias.shape[0] == out_channels, "bias must have same number of channels as output"
    
    # Launch kernel
    _conv2d_kernel[1](input, weight, output, bias, 
                      iH, iW, oH, oW, 
                      in_channels, out_channels, 
                      kH, kW, 
                      strideH, strideW, 
                      padH, padW, 
                      dilationH, dilationW, 
                      groups)
    
    return output

@triton.jit
def _conv2d_kernel(input_ptr, weight_ptr, output_ptr, bias_ptr, 
                   iH: tl.constexpr, iW: tl.constexpr, oH: tl.constexpr, oW: tl.constexpr,
                   in_channels: tl.constexpr, out_channels: tl.constexpr,
                   kH: tl.constexpr, kW: tl.constexpr,
                   strideH: tl.constexpr, strideW: tl.constexpr,
                   padH: tl.constexpr, padW: tl.constexpr,
                   dilationH: tl.constexpr, dilationW: tl.constexpr,
                   groups: tl.constexpr):
    # Get block indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    
    # Calculate group size
    in_ch_per_group = in_channels // groups
    out_ch_per_group = out_channels // groups
    
    # Calculate group index
    group_idx = out_ch_idx // out_ch_per_group
    
    # Initialize output
    output = tl.zeros((oH, oW), dtype=tl.float32)
    
    # Loop over input channels
    for in_ch in range(in_ch_per_group):
        # Calculate input channel index within group
        in_ch_global = group_idx * in_ch_per_group + in_ch
        
        # Load weight
        weight = tl.load(weight_ptr + 
                        (out_ch_idx * in_ch_per_group + in_ch) * kH * kW + 
                        tl.arange(0, kH)[:, None] * kW + 
                        tl.arange(0, kW)[None, :])
        
        # Loop over output spatial positions
        for oh in range(oH):
            for ow in range(oW):
                # Calculate input spatial positions
                ih_start = oh * strideH - padH
                iw_start = ow * strideW - padW
                
                # Initialize accumulator
                acc = 0.0
                
                # Loop over kernel spatial positions
                for kh in range(kH):
                    for kw in range(kW):
                        # Calculate input position
                        ih = ih_start + kh * dilationH
                        iw = iw_start + kw * dilationW
                        
                        # Check bounds
                        if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                            # Load input
                            input_val = tl.load(input_ptr + 
                                               batch_idx * in_channels * iH * iW + 
                                               in_ch_global * iH * iW + 
                                               ih * iW + iw)
                            # Accumulate
                            acc += input_val * weight[kh, kw]
                
                # Store output
                tl.store(output_ptr + 
                        batch_idx * out_channels * oH * oW + 
                        out_ch_idx * oH * oW + 
                        oh * oW + ow, acc, mask=(oh < oH) & (ow < oW))
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + out_ch_idx)
        for oh in range(oH):
            for ow in range(oW):
                output_val = tl.load(output_ptr + 
                                    batch_idx * out_channels * oH * oW + 
                                    out_ch_idx * oH * oW + 
                                    oh * oW + ow)
                output_val += bias
                tl.store(output_ptr + 
                        batch_idx * out_channels * oH * oW + 
                        out_ch_idx * oH * oW + 
                        oh * oW + ow, output_val, mask=(oh < oH) & (ow < oW))