import torch
import triton
import triton.language as tl
import math

@triton.jit
def _conv2d_add_kernel(
    input_ptr, weight_ptr, bias_ptr, other_ptr, output_ptr,
    iH, iW, oH, oW, in_channels, out_channels, kH, kW,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, alpha,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)
    
    # Calculate output dimensions
    output_size = oH * oW * out_channels
    
    # Each thread handles one output element
    if pid >= output_size:
        return
    
    # Calculate which output element this thread is handling
    out_c = pid % out_channels
    out_h = (pid // out_channels) % oH
    out_w = (pid // out_channels) // oH
    
    # Calculate input indices
    in_h_start = out_h * stride_h - pad_h
    in_w_start = out_w * stride_w - pad_w
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over input channels and kernel elements
    for g in range(groups):
        for kh in range(kH):
            for kw in range(kW):
                # Calculate input indices with dilation
                ih = in_h_start + kh * dilation_h
                iw = in_w_start + kw * dilation_w
                
                # Check bounds
                if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                    # Calculate input and weight indices
                    in_c = (g * (in_channels // groups)) + (out_c % (out_channels // groups))
                    weight_idx = out_c * (in_channels // groups) * kH * kW + (g * (in_channels // groups) * kH * kW) + kh * kW + kw
                    
                    # Load input and weight
                    input_idx = batch_id * (in_channels * iH * iW) + in_c * (iH * iW) + ih * iW + iw
                    input_val = tl.load(input_ptr + input_idx, mask=True)
                    weight_val = tl.load(weight_ptr + weight_idx, mask=True)
                    acc += input_val * weight_val
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_idx = out_c
        bias_val = tl.load(bias_ptr + bias_idx, mask=True)
        acc += bias_val
    
    # Apply alpha scaling to other tensor
    if other_ptr is not None:
        other_idx = batch_id * (out_channels * oH * oW) + out_c * (oH * oW) + out_h * oW + out_w
        other_val = tl.load(other_ptr + other_idx, mask=True)
        acc += alpha * other_val
    
    # Store result
    output_idx = batch_id * (out_channels * oH * oW) + out_c * (oH * oW) + out_h * oW + out_w
    tl.store(output_ptr + output_idx, acc, mask=True)

def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
    # Handle scalar inputs
    if not torch.is_tensor(stride):
        stride = (stride, stride)
    if not torch.is_tensor(padding):
        padding = (padding, padding)
    if not torch.is_tensor(dilation):
        dilation = (dilation, dilation)
    
    # Get dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    
    oH = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    if out is not None:
        output = out
    else:
        output = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Handle other tensor
    if other is not None:
        if not torch.is_tensor(other):
            other = torch.tensor(other, device=input.device, dtype=input.dtype)
        if other.dim() == 0:
            other = other.expand(batch_size, out_channels, oH, oW)
        elif other.shape == (batch_size, out_channels, oH, oW):
            pass
        else:
            raise ValueError("other tensor must be scalar or have shape (batch_size, out_channels, oH, oW)")
    
    # Handle bias tensor
    if bias is not None:
        if bias.shape != (out_channels,):
            raise ValueError("bias tensor must have shape (out_channels,)")
    
    # Launch kernel
    grid_size = batch_size * out_channels * oH * oW
    block_size = 256
    
    # Calculate grid
    grid = (triton.cdiv(grid_size, block_size), 1)
    
    # Launch kernel
    _conv2d_add_kernel[grid](
        input, weight, bias, other, output,
        iH, iW, oH, oW, in_channels, out_channels, kH, kW,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
        groups, alpha,
        BLOCK_SIZE=block_size
    )
    
    return output

##################################################################################################################################################



def test_conv2d_add():
    results = {}

    # Test case 1: Basic convolution with bias, no addition
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    bias_tensor = torch.randn(2, device='cuda')
    results["test_case_1"] = conv2d_add(input_tensor, weight_tensor, bias=bias_tensor)

    # Test case 2: Convolution with addition of a scalar
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    scalar_addition = 2.0
    results["test_case_2"] = conv2d_add(input_tensor, weight_tensor, other=scalar_addition)

    # Test case 3: Convolution with addition of a tensor
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    other_tensor = torch.randn(1, 2, 3, 3, device='cuda')
    results["test_case_3"] = conv2d_add(input_tensor, weight_tensor, other=other_tensor)

    # Test case 4: Convolution with addition of a tensor and alpha scaling
    input_tensor = torch.randn(1, 3, 5, 5, device='cuda')
    weight_tensor = torch.randn(2, 3, 3, 3, device='cuda')
    other_tensor = torch.randn(1, 2, 3, 3, device='cuda')
    alpha_value = 0.5
    results["test_case_4"] = conv2d_add(input_tensor, weight_tensor, other=other_tensor, alpha=alpha_value)

    return results

test_results = test_conv2d_add()
