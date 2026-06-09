import torch
import triton
import triton.language as tl

@triton.jit
def relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    bias_stride_0,
    batch_size, in_channels, out_channels, iH, iW, oH, oW, kH, kW,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output indices
    batch_idx = pid // (out_channels * oH * oW)
    remaining = pid % (out_channels * oH * oW)
    out_ch_idx = remaining // (oH * oW)
    remaining = remaining % (oH * oW)
    out_h_idx = remaining // oW
    out_w_idx = remaining % oW
    
    # Shared memory for input tile and weight tile
    input_tile = tl.shared.load(input_ptr + batch_idx * input_stride_0 + out_ch_idx * input_stride_1 + out_h_idx * stride_h * input_stride_2 + out_w_idx * stride_w * input_stride_3, (kH, kW))
    weight_tile = tl.shared.load(weight_ptr + out_ch_idx * weight_stride_0 + out_ch_idx * weight_stride_1 + 0 * weight_stride_2 + 0 * weight_stride_3, (kH, kW))
    
    # Compute convolution
    acc = 0.0
    for kh in range(kH):
        for kw in range(kW):
            input_val = input_tile[kh, kw]
            weight_val = weight_tile[kh, kw]
            acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + out_ch_idx * bias_stride_0)
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Store result
    tl.store(output_ptr + batch_idx * output_stride_0 + out_ch_idx * output_stride_1 + out_h_idx * output_stride_2 + out_w_idx * output_stride_3, acc)

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    # Handle stride, padding, dilation as tuples
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Compute output dimensions
    oH = (iH + 2 * padding_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * padding_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    # Output tensor
    output = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=input.dtype)
    
    # Grid dimensions
    grid = (batch_size * out_channels * oH * oW, 1)
    
    # Launch kernel
    relu_conv2d_kernel[grid](
        input, weight, bias, output,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        bias.stride(0) if bias is not None else 0,
        batch_size, in_channels, out_channels, iH, iW, oH, oW, kH, kW,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16
    )
    
    return output
