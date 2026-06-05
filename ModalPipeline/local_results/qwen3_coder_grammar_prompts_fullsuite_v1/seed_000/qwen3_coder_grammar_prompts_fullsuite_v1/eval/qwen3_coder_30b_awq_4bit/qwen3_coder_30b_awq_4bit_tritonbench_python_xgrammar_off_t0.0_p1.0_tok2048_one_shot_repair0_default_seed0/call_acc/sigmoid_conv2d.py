import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, oH, oW,
    kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, group_size,
    BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16
):
    # Get the block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute the output indices
    m = pid_m * BLOCK_SIZE_M
    n = pid_n * BLOCK_SIZE_N
    k = pid_k * BLOCK_SIZE_K
    
    # Compute the output indices
    out_m = m // BLOCK_SIZE_M
    out_n = n // BLOCK_SIZE_N
    out_k = k // BLOCK_SIZE_K
    
    # Compute the output indices
    out_idx = out_m * BLOCK_SIZE_M + out_n * BLOCK_SIZE_N + out_k * BLOCK_SIZE_K
    
    # Compute the output indices
    if out_idx >= batch_size * out_channels * oH * oW:
        return
    
    # Compute the output indices
    batch_idx = out_idx // (out_channels * oH * oW)
    channel_idx = (out_idx // (oH * oW)) % out_channels
    h_idx = (out_idx // oW) % oH
    w_idx = out_idx % oW
    
    # Compute the output value
    output_val = 0.0
    for g in range(groups):
        for kh in range(kH):
            for kw in range(kW):
                # Compute the input indices
                ih = h_idx * stride_h + kh * dilation_h - pad_h
                iw = w_idx * stride_w + kw * dilation_w - pad_w
                
                # Check if the input indices are valid
                if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                    # Compute the weight indices
                    weight_idx = channel_idx * (in_channels // groups) * kH * kW + g * kH * kW + kh * kW + kw
                    
                    # Compute the input indices
                    input_idx = batch_idx * in_channels * iH * iW + g * (in_channels // groups) * iH * iW + ih * iW + iw
                    
                    # Compute the output value
                    output_val += tl.load(input_ptr + input_idx) * tl.load(weight_ptr + weight_idx)
    
    # Add bias if provided
    if bias_ptr is not None:
        output_val += tl.load(bias_ptr + channel_idx)
    
    # Apply sigmoid activation
    output_val = 1.0 / (1.0 + tl.exp(-output_val))
    
    # Store the output value
    output_idx = batch_idx * out_channels * oH * oW + channel_idx * oH * oW + h_idx * oW + w_idx
    tl.store(output_ptr + output_idx, output_val)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Ensure inputs are tensors
    input = torch.as_tensor(input)
    weight = torch.as_tensor(weight)
    
    # Get input dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Handle stride
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    # Handle padding
    if isinstance(padding, int):
        pad_h = pad_w = padding
    elif isinstance(padding, str):
        if padding == 'valid':
            pad_h = pad_w = 0
        elif padding == 'same':
            pad_h = (kH - 1) // 2
            pad_w = (kW - 1) // 2
        else:
            pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    # Handle dilation
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    
    # Compute output dimensions
    oH = (iH + 2 * pad_h - (kH - 1) * dilation_h - 1) // stride_h + 1
    oW = (iW + 2 * pad_w - (kW - 1) * dilation_w - 1) // stride_w + 1
    
    # Initialize output tensor
    if out is None:
        out = torch.empty(batch_size, out_channels, oH, oW, dtype=torch.float32, device=input.device)
    else:
        out = torch.as_tensor(out)
    
    # Prepare kernel launch parameters
    grid = (
        triton.cdiv(oH, 16),
        triton.cdiv(oW, 16),
        triton.cdiv(out_channels, 16)
    )
    
    # Launch kernel
    conv2d_sigmoid_kernel[grid](
        input,
        weight,
        bias,
        out,
        batch_size,
        in_channels,
        out_channels,
        iH,
        iW,
        oH,
        oW,
        kH,
        kW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        in_channels // groups
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
