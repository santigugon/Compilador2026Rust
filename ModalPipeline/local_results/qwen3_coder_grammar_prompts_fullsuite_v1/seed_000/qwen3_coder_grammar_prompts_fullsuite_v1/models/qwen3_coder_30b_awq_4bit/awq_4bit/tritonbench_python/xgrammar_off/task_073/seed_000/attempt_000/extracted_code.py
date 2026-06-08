import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, oH, oW,
    kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, in_channels_per_group, out_channels_per_group,
    BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16
):
    # Get the block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute the output indices
    output_m = pid_m * BLOCK_SIZE_M
    output_n = pid_n * BLOCK_SIZE_N
    output_k = pid_k * BLOCK_SIZE_K
    
    # Compute the input indices
    input_m = output_m
    input_n = output_n
    input_k = output_k
    
    # Load the input and weight tensors
    input_tile = tl.load(input_ptr + input_m * iW + input_n)
    weight_tile = tl.load(weight_ptr + output_k * kH * kW + input_k)
    
    # Compute the convolution
    output_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(kH):
        for j in range(kW):
            input_val = input_tile[input_m + i * dilation_h, input_n + j * dilation_w]
            weight_val = weight_tile[output_k + i * kW + j]
            output_tile += input_val * weight_val
    
    # Store the output
    tl.store(output_ptr + output_m * oW + output_n, output_tile)

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
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    
    oH = (iH + 2 * pad_h - (kH - 1) * dilation_h - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - (kW - 1) * dilation_w - 1) // stride_w + 1
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Handle groups
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    # Launch kernel
    grid = (
        triton.cdiv(oH, 16),
        triton.cdiv(oW, 16),
        triton.cdiv(out_channels, 16)
    )
    
    # Create a kernel launch configuration
    config = {
        'batch_size': batch_size,
        'in_channels': in_channels,
        'out_channels': out_channels,
        'iH': iH,
        'iW': iW,
        'oH': oH,
        'oW': oW,
        'kH': kH,
        'kW': kW,
        'stride_h': stride_h,
        'stride_w': stride_w,
        'pad_h': pad_h,
        'pad_w': pad_w,
        'dilation_h': dilation_h,
        'dilation_w': dilation_w,
        'groups': groups,
        'in_channels_per_group': in_channels_per_group,
        'out_channels_per_group': out_channels_per_group
    }
    
    # Launch kernel
    conv2d_kernel[grid](
        input,
        weight,
        bias,
        output,
        **config
    )
    
    return output
