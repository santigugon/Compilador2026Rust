import torch
import triton
import triton.language as tl
from torch.nn import functional as F

@triton.jit
def _conv2d_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape0, input_shape1, input_shape2, input_shape3,
    weight_shape0, weight_shape1, weight_shape2, weight_shape3,
    output_shape0, output_shape1, output_shape2, output_shape3,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    
    # Calculate output dimensions
    out_h = output_shape2
    out_w = output_shape3
    
    # Shared memory for input tile
    input_tile = tl.shared_ptr(input_ptr + batch_id * input_shape1 * input_shape2 * input_shape3, 
                              shape=(input_shape1, input_shape2, input_shape3), 
                              dtype=tl.float32)
    
    # Shared memory for weight tile
    weight_tile = tl.shared_ptr(weight_ptr + out_ch_id * weight_shape1 * weight_shape2 * weight_shape3, 
                               shape=(weight_shape1, weight_shape2, weight_shape3), 
                               dtype=tl.float32)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Calculate group-specific indices
        in_ch_start = g * (input_shape1 // groups)
        in_ch_end = (g + 1) * (input_shape1 // groups)
        weight_start = g * (weight_shape0 // groups) * weight_shape1 * weight_shape2 * weight_shape3
        
        # Loop over input channels
        for ch in range(in_ch_start, in_ch_end):
            # Load input tile
            input_offset = batch_id * input_shape1 * input_shape2 * input_shape3 + ch * input_shape2 * input_shape3
            input_tile = tl.load(input_ptr + input_offset, mask=tl.arange(0, BLOCK_SIZE) < input_shape2, other=0.0)
            
            # Load weight tile
            weight_offset = out_ch_id * weight_shape1 * weight_shape2 * weight_shape3 + (ch - in_ch_start) * weight_shape2 * weight_shape3
            weight_tile = tl.load(weight_ptr + weight_offset, mask=tl.arange(0, BLOCK_SIZE) < weight_shape2, other=0.0)
            
            # Perform convolution
            for i in range(out_h):
                for j in range(out_w):
                    # Calculate input indices with stride and padding
                    ih_start = i * stride_h - pad_h
                    iw_start = j * stride_w - pad_w
                    
                    # Convolution computation
                    for kh in range(weight_shape2):
                        for kw in range(weight_shape3):
                            ih = ih_start + kh * dilation_h
                            iw = iw_start + kw * dilation_w
                            
                            if ih >= 0 and ih < input_shape2 and iw >= 0 and iw < input_shape3:
                                input_val = tl.load(input_ptr + input_offset + ih * input_shape3 + iw, other=0.0)
                                weight_val = tl.load(weight_ptr + weight_offset + kh * weight_shape3 + kw, other=0.0)
                                acc[i, j] += input_val * weight_val
    
    # Apply bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_id, other=0.0)
        acc += bias_val
    
    # Apply sigmoid
    acc = 1.0 / (1.0 + tl.exp(-acc))
    
    # Store result
    output_offset = batch_id * output_shape1 * output_shape2 * output_shape3 + out_ch_id * output_shape2 * output_shape3
    tl.store(output_ptr + output_offset, acc, mask=tl.arange(0, BLOCK_SIZE) < output_shape2)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Handle scalar inputs
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    
    # Apply convolution using PyTorch's conv2d
    conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Apply sigmoid using Triton
    out_shape = conv_output.shape
    out = torch.empty_like(conv_output)
    
    # Create a simple elementwise sigmoid kernel for the output
    @triton.jit
    def _sigmoid_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + offsets, y, mask=mask)
    
    n = conv_output.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _sigmoid_kernel[grid](conv_output, out, n, BLOCK=block)
    
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
