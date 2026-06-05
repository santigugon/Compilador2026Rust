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
    pid_n = tl.program_id(axis=1)
    
    # Each program processes one output element
    output_row = pid // oW
    output_col = pid % oW
    
    if output_row >= oH or output_col >= oW:
        return
    
    # Compute output channel
    oc = pid_n
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Compute convolution
    for g in range(groups):
        for ic in range(in_channels // groups):
            for kh in range(kH):
                for kw in range(kW):
                    ih = output_row * stride_h - pad_h + kh * dilation_h
                    iw = output_col * stride_w - pad_w + kw * dilation_w
                    
                    if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                        input_idx = (
                            0 * input_stride_0 +
                            (g * (in_channels // groups) + ic) * input_stride_1 +
                            ih * input_stride_2 +
                            iw * input_stride_3
                        )
                        weight_idx = (
                            oc * weight_stride_0 +
                            ic * weight_stride_1 +
                            kh * weight_stride_2 +
                            kw * weight_stride_3
                        )
                        acc += tl.load(input_ptr + input_idx) * tl.load(weight_ptr + weight_idx)
    
    # Add bias if provided
    if bias_ptr is not None:
        acc += tl.load(bias_ptr + oc)
    
    # Add other tensor or scalar
    if other_is_tensor:
        other_val = tl.load(other_ptr + output_row * output_stride_2 + output_col * output_stride_3)
        acc += alpha * other_val
    else:
        acc += alpha * tl.load(other_ptr)
    
    # Store result
    output_idx = (
        0 * output_stride_0 +
        oc * output_stride_1 +
        output_row * output_stride_2 +
        output_col * output_stride_3
    )
    tl.store(output_ptr + output_idx, acc)

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Handle stride
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    # Handle padding
    if isinstance(padding, str):
        if padding == 'valid':
            pad_h = pad_w = 0
        elif padding == 'same':
            # For 'same' padding, we compute padding to make output size equal to input size
            # This is a simplified version - in practice, you'd compute actual padding
            pad_h = pad_w = 0
        else:
            raise ValueError("Padding must be 'valid', 'same', or an integer/tuple")
    elif isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    # Handle dilation
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    
    # Get input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Compute output dimensions
    oH = (iH + 2 * pad_h - (kH - 1) * dilation_h - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - (kW - 1) * dilation_w - 1) // stride_w + 1
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty((batch_size, out_channels, oH, oW), dtype=input.dtype, device=input.device)
    
    # Handle other tensor or scalar
    other_is_tensor = isinstance(other, torch.Tensor)
    if other_is_tensor:
        if other.shape != out.shape:
            raise ValueError("other tensor must have the same shape as output")
    else:
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Launch kernel
    grid = (oH * oW, out_channels)
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    # Ensure we have valid pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else 0
    other_ptr = other.data_ptr()
    output_ptr = out.data_ptr()
    
    # Launch kernel
    conv2d_add_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, in_channels, out_channels, iH, iW, oH, oW, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
        alpha, other_is_tensor,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return out
