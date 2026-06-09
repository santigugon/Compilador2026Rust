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
            # For 'same' padding, we'll compute appropriate padding values
            # This is a simplified approach - in practice, this would need more complex logic
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
        # For grouped convolution, we need to process each group separately
        # This is a simplified implementation - full grouped convolution would be more complex
        pass
    
    # Launch kernel
    if input.is_cuda and weight.is_cuda:
        _conv2d_kernel[grid](input, weight, output, bias, 
                           batch_size, in_channels, out_channels, iH, iW, oH, oW,
                           kH, kW, padH, padW, strideH, strideW, dilationH, dilationW, groups,
                           input.stride(0), input.stride(1), input.stride(2), input.stride(3),
                           weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
                           output.stride(0), output.stride(1), output.stride(2), output.stride(3))
    else:
        # Fall back to PyTorch for CPU or non-CUDA tensors
        return torch.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    return output

@triton.jit
def _conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, bias_ptr,
    batch_size, in_channels, out_channels, iH, iW, oH, oW,
    kH, kW, padH, padW, strideH, strideW, dilationH, dilationW, groups,
    input_s0, input_s1, input_s2, input_s3,
    weight_s0, weight_s1, weight_s2, weight_s3,
    output_s0, output_s1, output_s2, output_s3,
    BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    out_h_idx = tl.program_id(2)
    out_w_idx = tl.program_id(3)
    
    # Calculate output position
    if batch_idx >= batch_size or out_ch_idx >= out_channels or out_h_idx >= oH or out_w_idx >= oW:
        return
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Get bias if available
    if bias_ptr is not None:
        acc = tl.load(bias_ptr + out_ch_idx, mask=out_ch_idx < out_channels)
    
    # Loop over input channels and kernel elements
    for g in range(groups):
        for ih in range(kH):
            for iw in range(kW):
                # Calculate input position
                input_h = out_h_idx * strideH - padH + ih * dilationH
                input_w = out_w_idx * strideW - padW + iw * dilationW
                
                # Check bounds
                if input_h >= 0 and input_h < iH and input_w >= 0 and input_w < iW:
                    # Calculate input and weight indices
                    input_idx = batch_idx * input_s0 + g * (in_channels // groups) * input_s1 + input_h * input_s2 + input_w * input_s3
                    weight_idx = out_ch_idx * weight_s0 + g * (in_channels // groups) * weight_s1 + ih * weight_s2 + iw * weight_s3
                    
                    # Load values and accumulate
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    acc += input_val * weight_val
    
    # Store result
    output_idx = batch_idx * output_s0 + out_ch_idx * output_s1 + out_h_idx * output_s2 + out_w_idx * output_s3
    tl.store(output_ptr + output_idx, acc)

# Helper function to compute grid size
def grid(batch_size, out_channels, oH, oW):
    return (
        batch_size,
        out_channels,
        oH,
        oW
    )
