import torch
import triton
import triton.language as tl
import math

@triton.jit
def _conv2d_add_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    iH, iW, oH, oW, in_channels, out_channels, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)
    
    # Calculate output dimensions
    output_size = oH * oW * out_channels
    
    # Each thread handles one output element
    if pid >= output_size:
        return
    
    # Calculate which output element this thread handles
    out_c = pid % out_channels
    out_h = (pid // out_channels) % oH
    out_w = (pid // out_channels) // oH
    
    # Calculate input indices
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over input channels and kernel elements
    for g in range(groups):
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input indices with dilation
                ih = in_h_start + kh * dilation_h
                iw = in_w_start + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                    # Calculate input and weight indices
                    in_c = (g * (in_channels // groups)) + (out_c % (out_channels // groups))
                    weight_idx = out_c * (in_channels // groups) * kH * kW + (g * (in_channels // groups)) * kH * kW + kh * kW + kw
                    
                    # Load input and weight
                    input_val = tl.load(input_ptr + batch_id * (in_channels * iH * iW) + 
                                       g * (in_channels // groups) * iH * iW + 
                                       in_c * iH * iW + ih * iW + iw)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    acc += input_val * weight_val
    
    # Apply bias if present
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + out_c)
    
    # Add other tensor scaled by alpha
    if other_ptr is not None:
        other_val = tl.load(other_ptr + batch_id * (out_channels * oH * oW) + 
                           out_c * oH * oW + out_h * oW + out_w)
        acc += alpha * other_val
    
    # Store result
    tl.store(output_ptr + batch_id * (out_channels * oH * oW) + 
             out_c * oH * oW + out_h * oW + out_w, acc)

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Handle scalar inputs
    if not torch.is_tensor(stride):
        stride = (stride, stride)
    if not torch.is_tensor(padding):
        padding = (padding, padding)
    if not torch.is_tensor(dilation):
        dilation = (dilation, dilation)
    
    # Get dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    
    # Handle padding options
    if isinstance(padding, str):
        if padding == 'valid':
            pad_h, pad_w = 0, 0
        elif padding == 'same':
            # For 'same' padding, we need to calculate padding to maintain spatial dimensions
            # This is a simplified version - in practice, you'd want to compute the exact padding
            pad_h, pad_w = 0, 0
    
    # Calculate output height and width
    oH = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    if out is not None:
        output = out
    else:
        output = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Handle other tensor
    if other is not None:
        if not torch.is_tensor(other):
            other = torch.tensor(other, device=input.device, dtype=input.dtype)
        if other.dim() == 0:
            other = other.expand(batch_size, out_channels, oH, oW)
        elif other.dim() == 1:
            other = other.expand(batch_size, out_channels, oH, oW)
        elif other.dim() == 2:
            other = other.expand(batch_size, out_channels, oH, oW)
        elif other.dim() == 3:
            other = other.expand(batch_size, out_channels, oH, oW)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    other_ptr = other.data_ptr() if other is not None else None
    output_ptr = output.data_ptr()
    
    # Launch kernel
    grid = (oH * oW * out_channels, batch_size)
    BLOCK_SIZE = 256
    
    # Ensure we have the right number of groups
    if in_channels % groups != 0 or out_channels % groups != 0:
        raise ValueError("groups must divide both in_channels and out_channels")
    
    # Launch kernel
    _conv2d_add_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
        iH, iW, oH, oW, in_channels, out_channels, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        groups, alpha, BLOCK_SIZE
    )
    
    return output
