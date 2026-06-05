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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(out_channels, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(batch_size * oH * oW, BLOCK_SIZE_M)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_base = input_ptr
    weight_base = weight_ptr
    bias_base = bias_ptr
    output_base = output_ptr
    
    m_mask = offs_m < batch_size * oH * oW
    n_mask = offs_n < out_channels
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for g in range(groups):
        group_offset = g * (in_channels // groups)
        weight_offset = g * (out_channels // groups) * (in_channels // groups) * kH * kW
        for k in range(0, (in_channels // groups) * kH * kW, BLOCK_SIZE_K):
            k_mask = (offs_k + k) < (in_channels // groups) * kH * kW
            
            input_offset = (offs_m // (oH * oW)) * input_stride_0 + \
                          (offs_m % (oH * oW)) // oW * input_stride_2 + \
                          (offs_m % (oH * oW)) % oW * input_stride_3 + \
                          group_offset * input_stride_1
            
            weight_offset_local = (offs_n % (out_channels // groups)) * weight_stride_0 + \
                                 (offs_k + k) * weight_stride_3
            
            input_ptrs = input_base + input_offset
            weight_ptrs = weight_base + weight_offset_local
            
            input_vals = tl.load(input_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            weight_vals = tl.load(weight_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            
            accumulator += tl.dot(input_vals, weight_vals)
    
    if bias_ptr is not None:
        bias_offset = offs_n * bias_stride_0
        bias_ptrs = bias_base + bias_offset
        bias_vals = tl.load(bias_ptrs, mask=n_mask, other=0.0)
        accumulator += bias_vals[None, :]
    
    output_offset = (offs_m // (oH * oW)) * output_stride_0 + \
                   (offs_m % (oH * oW)) // oW * output_stride_2 + \
                   (offs_m % (oH * oW)) % oW * output_stride_3 + \
                   (offs_n // (out_channels // groups)) * output_stride_1
    
    output_ptrs = output_base + output_offset
    output_vals = tl.where(accumulator > 0, accumulator, 0.0)
    tl.store(output_ptrs, output_vals, mask=m_mask[:, None] & n_mask[None, :])

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    oH = (iH + 2 * padding_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * padding_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    output = torch.empty((batch_size, out_channels, oH, oW), dtype=input.dtype, device=input.device)
    
    if inplace:
        output = input
        # In-place ReLU would require a separate kernel or additional logic
        # For now, we'll compute the result in a new tensor and copy back if needed
        # This is a simplified approach that doesn't actually perform in-place
        # operation on the input tensor
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    num_warps = 4
    
    grid = lambda META: (
        triton.cdiv(batch_size * oH * oW, META['BLOCK_SIZE_M']) *
        triton.cdiv(out_channels, META['BLOCK_SIZE_N']),
    )
    
    relu_conv2d_kernel[grid](
        input, weight, bias, output,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        bias.stride(0) if bias is not None else 0,
        batch_size, in_channels, out_channels, iH, iW, oH, oW, kH, kW,
        stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=num_warps
    )
    
    return output
