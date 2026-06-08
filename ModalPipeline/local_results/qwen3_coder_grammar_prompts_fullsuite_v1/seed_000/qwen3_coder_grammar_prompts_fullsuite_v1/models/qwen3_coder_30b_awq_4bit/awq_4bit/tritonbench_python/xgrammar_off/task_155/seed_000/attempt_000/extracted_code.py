import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_add_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, kH, kW,
    oH, oW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, alpha,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    batch_idx = pid_m // (out_channels * oH * oW)
    out_ch_idx = (pid_m // (oH * oW)) % out_channels
    oh_idx = (pid_m // oW) % oH
    ow_idx = pid_m % oW
    
    if batch_idx >= batch_size or out_ch_idx >= out_channels:
        return
    
    # Calculate output value
    output_val = 0.0
    
    # Loop over groups
    for g in range(groups):
        group_in_ch = in_channels // groups
        group_out_ch = out_channels // groups
        
        if out_ch_idx < g * group_out_ch or out_ch_idx >= (g + 1) * group_out_ch:
            continue
            
        # Loop over input channels
        for ic in range(group_in_ch):
            # Loop over kernel
            for kh in range(kH):
                for kw in range(kW):
                    ih = oh_idx * stride_h - pad_h + kh * dilation_h
                    iw = ow_idx * stride_w - pad_w + kw * dilation_w
                    
                    if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                        input_val = tl.load(input_ptr + 
                            batch_idx * (in_channels * iH * iW) +
                            (g * group_in_ch + ic) * (iH * iW) +
                            ih * iW + iw)
                        
                        weight_val = tl.load(weight_ptr + 
                            out_ch_idx * (group_in_ch * kH * kW) +
                            ic * (kH * kW) +
                            kh * kW + kw)
                        
                        output_val += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        output_val += tl.load(bias_ptr + out_ch_idx)
    
    # Add other tensor or scalar
    if other_ptr is not None:
        other_val = tl.load(other_ptr + 
            batch_idx * (out_channels * oH * oW) +
            out_ch_idx * (oH * oW) +
            oh_idx * oW + ow_idx)
        output_val += alpha * other_val
    
    # Store result
    tl.store(output_ptr + 
        batch_idx * (out_channels * oH * oW) +
        out_ch_idx * (oH * oW) +
        oh_idx * oW + ow_idx, output_val)

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Handle scalar inputs
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # Unpack dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    oH = (iH + 2 * padding[0] - (dilation[0] * (kH - 1) + 1)) // stride[0] + 1
    oW = (iW + 2 * padding[1] - (dilation[1] * (kW - 1) + 1)) // stride[1] + 1
    
    # Prepare output tensor
    if out is None:
        out = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Prepare other tensor if needed
    if other is not None:
        if isinstance(other, (int, float)):
            other = torch.full_like(input, other, device=input.device, dtype=input.dtype)
        if other.shape != out.shape:
            raise ValueError("other tensor must have the same shape as output")
    
    # Prepare bias tensor if needed
    if bias is not None:
        if bias.shape != (out_channels,):
            raise ValueError("bias tensor must have shape (out_channels,)")
    
    # Launch kernel
    grid = (
        batch_size * out_channels * oH * oW,
        1,
        1
    )
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Launch kernel
    conv2d_add_kernel[grid](
        input,
        weight,
        bias,
        other,
        out,
        batch_size,
        in_channels,
        out_channels,
        iH,
        iW,
        kH,
        kW,
        oH,
        oW,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        alpha,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out
