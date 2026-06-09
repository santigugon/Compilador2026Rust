import torch
import triton
import triton.language as tl

@triton.jit
def _conv2d_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    weight_stride_0, weight_stride_1, weight_stride_2, weight_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    batch_size, in_channels, out_channels, iH, iW, oH, oW, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    batch_idx = pid // (out_channels * oH * oW)
    remaining = pid % (out_channels * oH * oW)
    out_ch_idx = remaining // (oH * oW)
    remaining = remaining % (oH * oW)
    out_h_idx = remaining // oW
    out_w_idx = remaining % oW
    
    if batch_idx >= batch_size or out_ch_idx >= out_channels or out_h_idx >= oH or out_w_idx >= oW:
        return
    
    acc = tl.zeros((1,), dtype=tl.float32)
    
    for g in range(groups):
        group_in_ch = in_channels // groups
        group_out_ch = out_channels // groups
        
        if out_ch_idx < g * group_out_ch or out_ch_idx >= (g + 1) * group_out_ch:
            continue
            
        for kh in range(kH):
            for kw in range(kW):
                ih = out_h_idx * stride_h - pad_h + kh * dilation_h
                iw = out_w_idx * stride_w - pad_w + kw * dilation_w
                
                if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                    for ic in range(group_in_ch):
                        input_val = tl.load(input_ptr + 
                                            batch_idx * input_stride_0 + 
                                            (g * group_in_ch + ic) * input_stride_1 + 
                                            ih * input_stride_2 + 
                                            iw * input_stride_3)
                        
                        weight_val = tl.load(weight_ptr + 
                                             out_ch_idx * weight_stride_0 + 
                                             ic * weight_stride_1 + 
                                             kh * weight_stride_2 + 
                                             kw * weight_stride_3)
                        
                        acc += input_val * weight_val
    
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_idx)
        acc += bias_val
    
    output_val = tl.sigmoid(acc)
    
    tl.store(output_ptr + 
             batch_idx * output_stride_0 + 
             out_ch_idx * output_stride_1 + 
             out_h_idx * output_stride_2 + 
             out_w_idx * output_stride_3, 
             output_val)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
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
        output = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=torch.float32)
    else:
        output = out
    
    if groups > 1:
        assert in_channels % groups == 0 and out_channels % groups == 0
    
    grid = (batch_size * out_channels * oH * oW, 1)
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    _conv2d_sigmoid_kernel[grid](
        input, weight, bias, output,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        batch_size, in_channels, out_channels, iH, iW, oH, oW, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output
