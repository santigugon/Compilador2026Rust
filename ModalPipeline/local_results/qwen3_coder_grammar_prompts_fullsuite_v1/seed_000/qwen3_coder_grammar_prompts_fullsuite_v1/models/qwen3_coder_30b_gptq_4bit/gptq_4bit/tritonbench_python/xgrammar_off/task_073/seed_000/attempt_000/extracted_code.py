import torch
import triton
import triton.language as tl
import math

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    iH, iW, oH, oW, in_channels, out_channels, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_C: tl.constexpr
):
    # Get program ID
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Calculate output dimensions
    output_h = oH
    output_w = oW
    
    # Calculate block start positions
    start_h = pid_h * BLOCK_SIZE_H
    start_w = pid_w * BLOCK_SIZE_W
    start_c = pid_c * BLOCK_SIZE_C
    
    # Create masks for valid output positions
    mask_h = start_h + tl.arange(0, BLOCK_SIZE_H) < output_h
    mask_w = start_w + tl.arange(0, BLOCK_SIZE_W) < output_w
    mask_c = start_c + tl.arange(0, BLOCK_SIZE_C) < out_channels
    
    # Load bias if available
    bias = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)
    if bias_ptr is not None:
        bias_offsets = start_c + tl.arange(0, BLOCK_SIZE_C)
        bias = tl.load(bias_ptr + bias_offsets, mask=mask_c, other=0.0)
    
    # Initialize output
    output = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    for g in range(groups):
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input positions
                input_h_start = start_h * stride_h - pad_h + kh * dilation_h
                input_w_start = start_w * stride_w - pad_w + kw * dilation_w
                
                # Calculate valid input region
                valid_h_start = tl.maximum(0, input_h_start)
                valid_h_end = tl.minimum(iH, input_h_start + BLOCK_SIZE_H * stride_h)
                valid_w_start = tl.maximum(0, input_w_start)
                valid_w_end = tl.minimum(iW, input_w_start + BLOCK_SIZE_W * stride_w)
                
                # Load input data
                input_offsets_h = valid_h_start + tl.arange(0, BLOCK_SIZE_H) * stride_h
                input_offsets_w = valid_w_start + tl.arange(0, BLOCK_SIZE_W) * stride_w
                
                # Create mask for valid input positions
                input_mask_h = input_offsets_h >= 0 & input_offsets_h < iH
                input_mask_w = input_offsets_w >= 0 & input_offsets_w < iW
                
                # Load weight data
                weight_offset = g * (out_channels // groups) + tl.arange(0, BLOCK_SIZE_C)
                weight = tl.load(weight_ptr + weight_offset + kh * kW + kw, mask=mask_c, other=0.0)
                
                # Perform convolution
                for h in range(BLOCK_SIZE_H):
                    for w in range(BLOCK_SIZE_W):
                        if input_mask_h[h] and input_mask_w[w]:
                            input_val = tl.load(input_ptr + input_offsets_h[h] * iW + input_offsets_w[w], mask=True)
                            output[h, w, :] += input_val * weight
    
    # Add bias and store result
    output = output + bias
    output_offsets_h = start_h + tl.arange(0, BLOCK_SIZE_H)
    output_offsets_w = start_w + tl.arange(0, BLOCK_SIZE_W)
    output_offsets_c = start_c + tl.arange(0, BLOCK_SIZE_C)
    
    # Store output
    for h in range(BLOCK_SIZE_H):
        for w in range(BLOCK_SIZE_W):
            for c in range(BLOCK_SIZE_C):
                if mask_h[h] and mask_w[w] and mask_c[c]:
                    output_ptr[(output_offsets_h[h] * oW + output_offsets_w[w]) * out_channels + output_offsets_c[c]] = output[h, w, c]

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Handle scalar inputs
    if not isinstance(stride, (tuple, list)):
        stride = (stride, stride)
    if not isinstance(padding, (tuple, list)):
        padding = (padding, padding)
    if not isinstance(dilation, (tuple, list)):
        dilation = (dilation, dilation)
    
    # Extract dimensions
    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    
    oH = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(minibatch, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Handle groups
    if in_channels % groups != 0 or out_channels % groups != 0:
        raise ValueError("in_channels and out_channels must be divisible by groups")
    
    # Set block sizes
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    
    # Calculate grid dimensions
    grid_h = math.ceil(oH / BLOCK_SIZE_H)
    grid_w = math.ceil(oW / BLOCK_SIZE_W)
    grid_c = math.ceil(out_channels / BLOCK_SIZE_C)
    
    # Launch kernel
    grid = (grid_h, grid_w, grid_c)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = output.data_ptr()
    
    # Call kernel
    _conv2d_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        iH, iW, oH, oW, in_channels, out_channels, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )
    
    return output
