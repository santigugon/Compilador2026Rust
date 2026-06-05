import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, oH, oW,
    kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, in_channels_per_group, out_channels_per_group,
    input_batch_stride, input_channel_stride, input_h_stride, input_w_stride,
    weight_batch_stride, weight_channel_stride, weight_h_stride, weight_w_stride,
    output_batch_stride, output_channel_stride, output_h_stride, output_w_stride,
    bias_stride,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(out_channels, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(oH * oW, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    output_offset = pid_m * output_h_stride + pid_n * output_channel_stride
    output_ptrs = output_ptr + output_offset
    
    for group in range(groups):
        input_group_offset = group * in_channels_per_group * input_channel_stride
        weight_group_offset = group * out_channels_per_group * weight_batch_stride
        bias_group_offset = group * out_channels_per_group * bias_stride
        
        for k in range(0, in_channels_per_group * kH * kW, BLOCK_SIZE_K):
            input_ptrs = input_ptr + input_group_offset
            weight_ptrs = weight_ptr + weight_group_offset
            
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            
            for i in range(0, in_channels_per_group * kH * kW, BLOCK_SIZE_K):
                input_ptrs = input_ptr + input_group_offset + i * input_channel_stride
                weight_ptrs = weight_ptr + weight_group_offset + i * weight_batch_stride
                
                input_ptrs = input_ptrs + (offs_m[:, None] * stride_h - pad_h) * input_h_stride + (offs_k[None, :] * dilation_h) * input_w_stride
                weight_ptrs = weight_ptrs + (offs_n[:, None] * weight_batch_stride) + (offs_k[None, :] * weight_batch_stride)
                
                input_vals = tl.load(input_ptrs, mask=(offs_m[:, None] < oH) & (offs_k[None, :] < in_channels_per_group * kH * kW), other=0.0)
                weight_vals = tl.load(weight_ptrs, mask=(offs_n[:, None] < out_channels_per_group) & (offs_k[None, :] < in_channels_per_group * kH * kW), other=0.0)
                
                acc += tl.dot(input_vals, weight_vals)
            
            output_ptrs = output_ptr + output_offset + (offs_m[:, None] * output_h_stride) + (offs_n[None, :] * output_channel_stride)
            tl.store(output_ptrs, acc, mask=(offs_m[:, None] < oH) & (offs_n[None, :] < out_channels_per_group))

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    
    out_h = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    out_w = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    output = torch.empty((batch_size, out_channels, out_h, out_w), device=input.device, dtype=input.dtype)
    
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    num_warps = 4
    
    grid = (triton.cdiv(out_h * out_w, BLOCK_SIZE_M) * triton.cdiv(out_channels, BLOCK_SIZE_N),)
    
    input_strides = input.stride()
    weight_strides = weight.stride()
    output_strides = output.stride()
    
    bias_stride = 1 if bias is not None else 0
    
    conv2d_kernel[grid](
        input, weight, bias, output,
        batch_size, in_channels, out_channels, iH, iW, out_h, out_w,
        kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        groups, in_channels_per_group, out_channels_per_group,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
        bias_stride if bias is not None else 0,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=num_warps
    )
    
    return output
