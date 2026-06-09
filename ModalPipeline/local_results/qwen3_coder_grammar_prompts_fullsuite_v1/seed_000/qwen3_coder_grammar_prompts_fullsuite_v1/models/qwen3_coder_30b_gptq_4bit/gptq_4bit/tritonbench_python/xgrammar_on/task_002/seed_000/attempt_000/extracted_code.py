import torch
import triton
import triton.language as tl

def _get_padding(padding):
    if isinstance(padding, str):
        if padding == 'valid':
            return (0, 0)
        elif padding == 'same':
            return (0, 0)
        else:
            raise ValueError(f"Unknown padding string: {padding}")
    elif isinstance(padding, int):
        return (padding, padding)
    elif isinstance(padding, tuple):
        return padding
    else:
        raise ValueError(f"Invalid padding type: {type(padding)}")

def _get_stride(stride):
    if isinstance(stride, int):
        return (stride, stride)
    elif isinstance(stride, tuple):
        return stride
    else:
        raise ValueError(f"Invalid stride type: {type(stride)}")

def _get_dilation(dilation):
    if isinstance(dilation, int):
        return (dilation, dilation)
    elif isinstance(dilation, tuple):
        return dilation
    else:
        raise ValueError(f"Invalid dilation type: {type(dilation)}")

@triton.jit
def _conv2d_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    iH, iW, oH, oW,
    in_channels, out_channels, kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Calculate output dimensions
    output_h = oH
    output_w = oW
    
    # Calculate block offsets
    block_h_start = pid_h * BLOCK_SIZE_H
    block_w_start = pid_w * BLOCK_SIZE_W
    block_c_start = pid_c * BLOCK_SIZE_C
    
    # Load bias if available
    bias = None
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + block_c_start, mask=block_c_start + tl.arange(0, BLOCK_SIZE_C) < out_channels)
    
    # Process output blocks
    for h in range(0, output_h, BLOCK_SIZE_H):
        for w in range(0, output_w, BLOCK_SIZE_W):
            # Calculate output indices
            out_h = h + tl.arange(0, BLOCK_SIZE_H)
            out_w = w + tl.arange(0, BLOCK_SIZE_W)
            
            # Create mask for valid output indices
            mask_h = out_h < output_h
            mask_w = out_w < output_w
            mask = mask_h[:, None] * mask_w[None, :]
            
            # Initialize output
            out = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
            
            # Perform convolution
            for kh in range(kH):
                for kw in range(kW):
                    # Calculate input indices
                    input_h = out_h * stride_h + kh * dilation_h - pad_h
                    input_w = out_w * stride_w + kw * dilation_w - pad_w
                    
                    # Load input
                    input_ptr_offset = input_ptr + (
                        (input_h[:, None] * iW + input_w[None, :]) * in_channels // groups
                    )
                    
                    # Load weight
                    weight_ptr_offset = weight_ptr + (
                        (kh * kW + kw) * (in_channels // groups)
                    )
                    
                    # Perform convolution
                    input_val = tl.load(input_ptr_offset, mask=mask, other=0.0)
                    weight_val = tl.load(weight_ptr_offset, mask=mask, other=0.0)
                    out += input_val * weight_val
            
            # Add bias
            if bias is not None:
                out += bias[None, :]
            
            # Apply sigmoid
            out = 1.0 / (1.0 + tl.exp(-out))
            
            # Store output
            output_ptr_offset = output_ptr + (
                (out_h[:, None] * oW + out_w[None, :]) * out_channels
            )
            tl.store(output_ptr_offset, out, mask=mask)


def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Validate inputs
    if input.dim() != 4:
        raise ValueError("input must be a 4D tensor")
    if weight.dim() != 4:
        raise ValueError("weight must be a 4D tensor")
    
    # Get dimensions
    minibatch, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Parse parameters
    stride_h, stride_w = _get_stride(stride)
    pad_h, pad_w = _get_padding(padding)
    dilation_h, dilation_w = _get_dilation(dilation)
    
    # Calculate output dimensions
    oH = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    if out is None:
        out = torch.empty(minibatch, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    else:
        if out.shape != (minibatch, out_channels, oH, oW):
            raise ValueError("out tensor has incorrect shape")
    
    # Handle bias
    if bias is not None:
        if bias.shape != (out_channels,):
            raise ValueError("bias tensor has incorrect shape")
    
    # Launch kernel
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    BLOCK_SIZE_C = 32
    
    grid = (
        triton.cdiv(oH, BLOCK_SIZE_H),
        triton.cdiv(oW, BLOCK_SIZE_W),
        triton.cdiv(out_channels, BLOCK_SIZE_C)
    )
    
    # Create pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = out.data_ptr()
    
    # Launch kernel
    _conv2d_sigmoid_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        iH, iW, oH, oW,
        in_channels, out_channels, kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        dilation_h, dilation_w,
        groups,
        BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C
    )
    
    return out