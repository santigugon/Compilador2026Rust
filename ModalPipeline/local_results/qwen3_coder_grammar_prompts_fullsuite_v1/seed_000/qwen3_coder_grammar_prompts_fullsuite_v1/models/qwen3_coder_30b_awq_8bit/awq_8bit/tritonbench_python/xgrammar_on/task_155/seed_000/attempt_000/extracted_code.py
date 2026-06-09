import torch
import triton
import triton.language as tl

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Handle scalar other
    if other is not None and not torch.is_tensor(other):
        alpha = alpha
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
        if out is not None:
            out = out + alpha * other
        else:
            out = input + alpha * other
        return out
    
    # Handle padding
    if isinstance(padding, str):
        if padding == 'valid':
            padding = (0, 0)
        elif padding == 'same':
            # For 'same' padding, we compute padding values
            # This is a simplified version - in practice, you'd compute exact padding
            padding = (0, 0)
        else:
            raise ValueError("Padding must be 'valid', 'same', or a tuple of integers")
    
    if isinstance(padding, int):
        padding = (padding, padding)
    
    if isinstance(stride, int):
        stride = (stride, stride)
    
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Get dimensions
    batch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Compute output dimensions
    padH, padW = padding
    strideH, strideW = stride
    dilationH, dilationW = dilation
    
    oH = (iH + 2 * padH - (dilationH * (kH - 1) + 1)) // strideH + 1
    oW = (iW + 2 * padW - (dilationW * (kW - 1) + 1)) // strideW + 1
    
    # Initialize output tensor
    if out is None:
        out = torch.empty((batch, out_channels, oH, oW), dtype=input.dtype, device=input.device)
    else:
        assert out.shape == (batch, out_channels, oH, oW), "Output tensor shape mismatch"
    
    # Handle bias
    if bias is not None:
        # Bias is of shape (out_channels,)
        bias = bias.view(1, out_channels, 1, 1)
    
    # Handle other tensor
    if other is not None:
        # Ensure other has the same shape as output
        if other.shape != out.shape:
            raise ValueError("other tensor must have the same shape as output")
        other = other * alpha
    
    # Launch kernel
    _conv2d_add_kernel(input, weight, bias, out, other, stride, padding, dilation, groups)
    
    return out

@triton.jit
def _conv2d_add_kernel(input_ptr, weight_ptr, bias_ptr, out_ptr, other_ptr, stride, padding, dilation, groups):
    # Get block indices
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Get input dimensions
    batch, in_channels, iH, iW = tl.load(input_ptr + 0).shape
    out_channels, _, kH, kW = tl.load(weight_ptr + 0).shape
    
    # Get stride, padding, dilation
    strideH, strideW = stride
    padH, padW = padding
    dilationH, dilationW = dilation
    
    # Get output dimensions
    oH = (iH + 2 * padH - (dilationH * (kH - 1) + 1)) // strideH + 1
    oW = (iW + 2 * padW - (dilationW * (kW - 1) + 1)) // strideW + 1
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    for g in range(groups):
        for kh in range(kH):
            for kw in range(kW):
                # Compute input indices
                ih = out_h_idx * strideH - padH + kh * dilationH
                iw = out_w_idx * strideW - padW + kw * dilationW
                
                # Check bounds
                if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                    # Load input
                    input_val = tl.load(input_ptr + batch_idx * (in_channels * iH * iW) + 
                                        (g * (in_channels // groups) + 0) * (iH * iW) + 
                                        ih * iW + iw)
                    # Load weight
                    weight_val = tl.load(weight_ptr + out_channel_idx * (in_channels // groups * kH * kW) + 
                                         (g * (in_channels // groups) + 0) * (kH * kW) + 
                                         kh * kW + kw)
                    acc += input_val * weight_val
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_channel_idx)
        acc += bias_val
    
    # Apply other tensor if present
    if other_ptr is not None:
        other_val = tl.load(other_ptr + batch_idx * (out_channels * oH * oW) + 
                            out_channel_idx * (oH * oW) + 
                            out_h_idx * oW + out_w_idx)
        acc += other_val
    
    # Store result
    tl.store(out_ptr + batch_idx * (out_channels * oH * oW) + 
             out_channel_idx * (oH * oW) + 
             out_h_idx * oW + out_w_idx, acc)