import torch
import triton
import triton.language as tl
from typing import Optional, Union, Tuple

@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    cdf = 0.5 * (1.0 + tl.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))
    output = x * cdf
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, oH, oW,
    kH, kW, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
    groups, group_size, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // (out_channels * oH * oW)
    remaining = pid % (out_channels * oH * oW)
    out_ch_id = remaining // (oH * oW)
    remaining = remaining % (oH * oW)
    oh = remaining // oW
    ow = remaining % oW

    if batch_id >= batch_size or out_ch_id >= out_channels or oh >= oH or ow >= oW:
        return

    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Loop over input channels and kernel elements
    for g in range(groups):
        for kh in range(kH):
            for kw in range(kW):
                ih = oh * stride_h - pad_h + kh * dilation_h
                iw = ow * stride_w - pad_w + kw * dilation_w
                
                if ih >= 0 and ih < iH and iw >= 0 and iw < iW:
                    for ic in range(group_size):
                        input_val = tl.load(input_ptr + 
                                            batch_id * (in_channels * iH * iW) +
                                            (g * group_size + ic) * (iH * iW) +
                                            ih * iW + iw)
                        weight_val = tl.load(weight_ptr + 
                                             out_ch_id * (groups * group_size * kH * kW) +
                                             g * (group_size * kH * kW) +
                                             ic * (kH * kW) + kh * kW + kw)
                        acc += input_val * weight_val
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_id)
        acc += bias_val
    
    # Store result
    output_ptr += batch_id * (out_channels * oH * oW) + out_ch_id * (oH * oW) + oh * oW + ow
    tl.store(output_ptr, acc)

def gelu_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    approximate: str = 'none',
    out: Optional[torch.Tensor] = None
) -> torch.Tensor:
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
            # For 'same' padding, we compute padding to ensure output size matches input size
            # This is a simplified version; actual implementation may vary
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
    oH = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1

    # Allocate output tensor
    if out is None:
        output = torch.empty(batch_size, out_channels, oH, oW, dtype=input.dtype, device=input.device)
    else:
        output = out

    # Compute group size
    group_size = in_channels // groups

    # Launch convolution kernel
    grid = (batch_size * out_channels * oH * oW,)
    BLOCK_SIZE = 1024
    conv2d_kernel[grid](
        input_ptr=input.data_ptr(),
        weight_ptr=weight.data_ptr(),
        bias_ptr=bias.data_ptr() if bias is not None else None,
        output_ptr=output.data_ptr(),
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        iH=iH,
        iW=iW,
        oH=oH,
        oW=oW,
        kH=kH,
        kW=kW,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=groups,
        group_size=group_size,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Apply GELU activation
    if approximate == 'none':
        # Use standard GELU
        output = torch.nn.functional.gelu(output)
    else:
        # Use approximate GELU
        output = torch.nn.functional.gelu(output, approximate=approximate)

    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, Tuple

# def gelu_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[int, Tuple[int, int]]=1, padding: Union[int, Tuple[int, int], str]=0, dilation: Union[int, Tuple[int, int]]=1, groups: int=1, approximate: str='none', out: Optional[Tensor]=None) -> Tensor:
#     conv_result = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.gelu(conv_result, approximate=approximate, out=out)

def test_gelu_conv2d():
    results = {}

    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = gelu_conv2d(input1, weight1)

    # Test case 2: Test with bias
    input2 = torch.randn(1, 3, 5, 5, device='cuda')
    weight2 = torch.randn(2, 3, 3, 3, device='cuda')
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = gelu_conv2d(input2, weight2, bias=bias2)

    # Test case 3: Test with stride and padding
    input3 = torch.randn(1, 3, 8, 8, device='cuda')
    weight3 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_3"] = gelu_conv2d(input3, weight3, stride=2, padding=1)

    # Test case 4: Test with dilation and groups
    input4 = torch.randn(1, 4, 10, 10, device='cuda')
    weight4 = torch.randn(4, 1, 3, 3, device='cuda')
    results["test_case_4"] = gelu_conv2d(input4, weight4, dilation=2, groups=4)

    return results

test_results = test_gelu_conv2d()
