import torch
import triton
import triton.language as tl

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Handle scalar inputs
    if not torch.is_tensor(other):
        alpha = alpha
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    if not torch.is_tensor(bias):
        bias = torch.tensor(bias, dtype=input.dtype, device=input.device) if bias is not None else None
    
    # Handle padding
    if isinstance(padding, str):
        if padding == 'valid':
            padding = (0, 0)
        elif padding == 'same':
            # For 'same' padding, we compute padding based on kernel size
            # This is a simplified version; in practice, you might want to compute
            # the exact padding needed for 'same' behavior
            padding = (0, 0)
        else:
            raise ValueError("Invalid padding string")
    
    # Handle stride and dilation
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    # Get dimensions
    batch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Compute output dimensions
    padH, padW = padding
    strideH, strideW = stride
    dilationH, dilationW = dilation
    
    # Calculate output height and width
    oH = (iH + 2 * padH - (dilationH * (kH - 1) + 1)) // strideH + 1
    oW = (iW + 2 * padW - (dilationW * (kW - 1) + 1)) // strideW + 1
    
    # Create output tensor
    if out is None:
        out = torch.empty(batch, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    else:
        assert out.shape == (batch, out_channels, oH, oW), "Output tensor shape mismatch"
    
    # Perform convolution
    _conv2d_kernel(input, weight, bias, out, batch, in_channels, out_channels, iH, iW, oH, oW, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, groups)
    
    # Add other tensor or scalar
    if other is not None:
        out = out + alpha * other
    
    return out

@triton.jit
def _conv2d_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, batch, in_channels, out_channels, iH, iW, oH, oW, kH, kW, strideH, strideW, padH, padW, dilationH, dilationW, groups):
    # Get block indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    
    # Calculate output dimensions
    output_size = oH * oW
    
    # Initialize output
    output = tl.zeros((oH, oW), dtype=tl.float32)
    
    # Loop over input channels
    for g in range(groups):
        # Calculate input channel offset
        in_ch_start = g * (in_channels // groups)
        in_ch_end = in_ch_start + (in_channels // groups)
        
        # Loop over kernel elements
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input indices
                ih = kh * dilationH - padH
                iw = kw * dilationW - padW
                
                # Load weight
                weight_val = tl.load(weight_ptr + out_ch_idx * (in_channels // groups) * kH * kW + g * kH * kW + kh * kW + kw)
                
                # Load input
                for oh in range(oH):
                    for ow in range(oW):
                        ih_idx = oh * strideH + ih
                        iw_idx = ow * strideW + iw
                        
                        # Check bounds
                        if ih_idx >= 0 and ih_idx < iH and iw_idx >= 0 and iw_idx < iW:
                            input_val = tl.load(input_ptr + batch_idx * in_channels * iH * iW + (in_ch_start + g) * iH * iW + ih_idx * iW + iw_idx)
                            output = output + input_val * weight_val

    # Store output
    for oh in range(oH):
        for ow in range(oW):
            tl.store(output_ptr + batch_idx * out_channels * oH * oW + out_ch_idx * oH * oW + oh * oW + ow, output[oh, ow])