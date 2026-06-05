import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_add_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    batch_size, in_channels, out_channels, iH, iW, oH, oW, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    alpha, other_is_tensor,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(oW, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(oH, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    m = tl.minimum(offs_m, oH)
    n = tl.minimum(offs_n, oW)
    
    output_ptrs = output_ptr + (
        (m[:, None] * output_stride_2 + n[None, :] * output_stride_3)
    )
    
    for g in range(groups):
        input_offset = g * (in_channels // groups) * iH * iW
        weight_offset = g * (out_channels // groups) * (in_channels // groups) * kH * kW
        output_offset = g * (out_channels // groups) * oH * oW
        
        for k in range(0, (in_channels // groups) * kH * kW, BLOCK_SIZE_K):
            input_ptrs = input_ptr + (
                (input_offset + (k // (kH * kW)) * iH * iW + 
                 (k % (kH * kW)) // kW * dilation_h + 
                 (k % (kH * kW)) % kW * dilation_w) * 
                input_stride_1 + 
                tl.arange(0, BLOCK_SIZE_M)[:, None] * input_stride_2 + 
                tl.arange(0, BLOCK_SIZE_N)[None, :] * input_stride_3
            )
            
            weight_ptrs = weight_ptr + (
                (weight_offset + (k // (kH * kW)) * kH * kW + 
                 (k % (kH * kW)) // kW * dilation_h + 
                 (k % (kH * kW)) % kW * dilation_w) * 
                weight_stride_1 + 
                tl.arange(0, BLOCK_SIZE_M)[:, None] * weight_stride_2 + 
                tl.arange(0, BLOCK_SIZE_N)[None, :] * weight_stride_3
            )
            
            input_vals = tl.load(input_ptrs, mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < oH) & 
                                               (tl.arange(0, BLOCK_SIZE_N)[None, :] < oW))
            weight_vals = tl.load(weight_ptrs, mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < kH) & 
                                                  (tl.arange(0, BLOCK_SIZE_N)[None, :] < kW))
            
            output_vals = tl.dot(input_vals, weight_vals)
            
            output_ptrs = output_ptr + (
                (output_offset + (m[:, None] * output_stride_2 + n[None, :] * output_stride_3)) * 
                output_stride_0
            )
            
            tl.store(output_ptrs, output_vals, mask=(tl.arange(0, BLOCK_SIZE_M)[:, None] < oH) & 
                                                 (tl.arange(0, BLOCK_SIZE_N)[None, :] < oW))

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
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
    
    oH = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    if out is None:
        out = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=input.dtype)
    
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    
    if other is not None:
        if isinstance(other, (int, float)):
            out = out + alpha * other
        else:
            out = out + alpha * other
    
    return out
