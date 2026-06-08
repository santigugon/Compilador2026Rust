import torch
import triton
import triton.language as tl
from typing import Union, Tuple, Optional

def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[str, int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1
) -> torch.Tensor:
    # Handle scalar inputs
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, str):
        if padding == 'valid':
            padding = (0, 0)
        elif padding == 'same':
            # For 'same' padding, we'll compute appropriate padding
            # This is a simplified version - in practice, you'd want to compute
            # the exact padding needed for same behavior
            padding = (0, 0)
        else:
            padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Get input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Compute output dimensions
    padH, padW = padding
    strideH, strideW = stride
    dilationH, dilationW = dilation
    
    # Calculate output height and width
    oH = (iH + 2 * padH - (dilationH * (kH - 1) + 1)) // strideH + 1
    oW = (iW + 2 * padW - (dilationW * (kW - 1) + 1)) // strideW + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Handle groups
    if groups > 1:
        # For grouped convolution, we process each group separately
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups
        
        for g in range(groups):
            start_in = g * in_channels_per_group
            end_in = (g + 1) * in_channels_per_group
            start_out = g * out_channels_per_group
            end_out = (g + 1) * out_channels_per_group
            
            # Extract group data
            input_group = input[:, start_in:end_in, :, :]
            weight_group = weight[start_out:end_out, :, :, :]
            
            # Compute convolution for this group
            _conv2d_group_kernel(
                input_group, weight_group, output[:, start_out:end_out, :, :], 
                bias[start_out:end_out] if bias is not None else None,
                strideH, strideW, padH, padW, dilationH, dilationW,
                batch_size, in_channels_per_group, iH, iW, 
                out_channels_per_group, kH, kW, oH, oW
            )
    else:
        # Standard convolution
        _conv2d_kernel(
            input, weight, output, bias,
            strideH, strideW, padH, padW, dilationH, dilationW,
            batch_size, in_channels, iH, iW, out_channels, kH, kW, oH, oW
        )
    
    return output

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    batch_size, in_channels, iH, iW, out_channels, kH, kW, oH, oW,
    BLOCK_SIZE: tl.constexpr = 128
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    
    # Shared memory for input tile
    tile_size = 16
    input_tile = tl.shared_ptr(tl.float32, tile_size * tile_size)
    
    # Initialize output
    out_val = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    for c in range(in_channels):
        for kh in range(kH):
            for kw in range(kW):
                # Compute input position
                ih = out_h_idx * stride_h + kh * dilation_h - pad_h
                iw = out_h_idx * stride_w + kw * dilation_w - pad_w
                
                # Check bounds
                if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                    # Load input value
                    input_val = tl.load(input_ptr + batch_idx * in_channels * iH * iW + 
                                       c * iH * iW + ih * iW + iw)
                    # Load weight value
                    weight_val = tl.load(weight_ptr + out_ch_idx * in_channels * kH * kW + 
                                        c * kH * kW + kh * kW + kw)
                    out_val += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        out_val += bias_val
    
    # Store output
    tl.store(output_ptr + batch_idx * out_channels * oH * oW + 
             out_ch_idx * oH * oW + out_h_idx * oW, out_val)

@triton.jit
def _conv2d_group_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    batch_size, in_channels, iH, iW, out_channels, kH, kW, oH, oW,
    BLOCK_SIZE: tl.constexpr = 128
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    
    # Initialize output
    out_val = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    for c in range(in_channels):
        for kh in range(kH):
            for kw in range(kW):
                # Compute input position
                ih = out_h_idx * stride_h + kh * dilation_h - pad_h
                iw = out_h_idx * stride_w + kw * dilation_w - pad_w
                
                # Check bounds
                if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                    # Load input value
                    input_val = tl.load(input_ptr + batch_idx * in_channels * iH * iW + 
                                       c * iH * iW + ih * iW + iw)
                    # Load weight value
                    weight_val = tl.load(weight_ptr + out_ch_idx * in_channels * kH * kW + 
                                        c * kH * kW + kh * kW + kw)
                    out_val += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        out_val += bias_val
    
    # Store output
    tl.store(output_ptr + batch_idx * out_channels * oH * oW + 
             out_ch_idx * oH * oW + out_h_idx * oW, out_val)
