import torch
import triton
import triton.language as tl
import math

@triton.jit
def _conv2d_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, oH, oW,
    kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, group_size,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    out_ch_idx = tl.program_id(1)
    
    # Calculate output dimensions
    output_h = oH
    output_w = oW
    
    # Load bias if available
    bias_val = tl.load(bias_ptr + out_ch_idx) if bias_ptr is not None else 0.0
    
    # Loop over output spatial dimensions
    for oh in range(output_h):
        for ow in range(output_w):
            # Initialize accumulator
            acc = 0.0
            
            # Loop over groups and input channels
            for g in range(groups):
                for ic in range(group_size):
                    # Calculate input indices
                    ih_start = oh * stride_h - pad_h
                    iw_start = ow * stride_w - pad_w
                    
                    # Loop over kernel dimensions
                    for kh in range(kH):
                        for kw in range(kW):
                            # Calculate input indices with dilation
                            ih = ih_start + kh * dilation_h
                            iw = iw_start + kw * dilation_w
                            
                            # Check bounds
                            if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                                # Calculate input and weight indices
                                input_idx = batch_idx * (in_channels * iH * iW) + \
                                           (g * group_size + ic) * (iH * iW) + \
                                           ih * iW + iw
                                weight_idx = out_ch_idx * (groups * group_size * kH * kW) + \
                                            (g * group_size + ic) * (kH * kW) + \
                                            kh * kW + kw
                                
                                # Load values and accumulate
                                input_val = tl.load(input_ptr + input_idx)
                                weight_val = tl.load(weight_ptr + weight_idx)
                                acc += input_val * weight_val
            
            # Add bias and apply sigmoid
            acc += bias_val
            sigmoid_val = 1.0 / (1.0 + tl.exp(-acc))
            
            # Store result
            output_idx = batch_idx * (out_channels * oH * oW) + \
                        out_ch_idx * (oH * oW) + \
                        oh * oW + ow
            tl.store(output_ptr + output_idx, sigmoid_val)

def _get_padding(padding, kernel_size):
    if isinstance(padding, str):
        if padding == 'valid':
            return 0, 0
        elif padding == 'same':
            return kernel_size // 2, kernel_size // 2
        else:
            raise ValueError(f"Unsupported padding string: {padding}")
    elif isinstance(padding, int):
        return padding, padding
    elif isinstance(padding, tuple):
        return padding[0], padding[1]
    else:
        raise ValueError(f"Unsupported padding type: {type(padding)}")

def _get_stride(stride):
    if isinstance(stride, int):
        return stride, stride
    elif isinstance(stride, tuple):
        return stride[0], stride[1]
    else:
        raise ValueError(f"Unsupported stride type: {type(stride)}")

def _get_dilation(dilation):
    if isinstance(dilation, int):
        return dilation, dilation
    elif isinstance(dilation, tuple):
        return dilation[0], dilation[1]
    else:
        raise ValueError(f"Unsupported dilation type: {type(dilation)}")

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Input shapes
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Handle stride
    stride_h, stride_w = _get_stride(stride)
    
    # Handle padding
    pad_h, pad_w = _get_padding(padding, kH)
    
    # Handle dilation
    dilation_h, dilation_w = _get_dilation(dilation)
    
    # Calculate output dimensions
    oH = (iH + 2 * pad_h - (kH - 1) * dilation_h - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - (kW - 1) * dilation_w - 1) // stride_w + 1
    
    # Handle groups
    if in_channels % groups != 0:
        raise ValueError("in_channels must be divisible by groups")
    group_size = in_channels // groups
    
    # Create output tensor
    if out is None:
        out = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    else:
        if out.shape != (batch_size, out_channels, oH, oW):
            raise ValueError("Output tensor has incorrect shape")
    
    # Prepare kernel launch parameters
    BLOCK_SIZE = 256
    grid = (batch_size, out_channels)
    
    # Launch kernel
    _conv2d_sigmoid_kernel[grid](
        input, weight, bias, out,
        batch_size, in_channels, out_channels, iH, iW, oH, oW,
        kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        groups, group_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
#     conv_result = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
#     result = torch.sigmoid(conv_result)
#     return result

def test_sigmoid_conv2d():
    results = {}

    # Test case 1: Basic test with no bias, stride, padding, dilation, or groups
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = sigmoid_conv2d(input1, weight1)

    # Test case 2: Test with bias
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = sigmoid_conv2d(input1, weight1, bias=bias2)

    # Test case 3: Test with stride
    results["test_case_3"] = sigmoid_conv2d(input1, weight1, stride=2)

    # Test case 4: Test with padding
    results["test_case_4"] = sigmoid_conv2d(input1, weight1, padding=1)

    return results

test_results = test_sigmoid_conv2d()
