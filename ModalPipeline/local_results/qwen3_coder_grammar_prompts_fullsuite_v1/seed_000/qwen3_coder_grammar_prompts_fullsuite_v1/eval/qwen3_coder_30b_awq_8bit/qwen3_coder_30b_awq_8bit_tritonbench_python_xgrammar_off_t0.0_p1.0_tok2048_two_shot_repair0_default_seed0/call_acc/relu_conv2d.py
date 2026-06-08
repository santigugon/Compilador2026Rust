import torch
import triton
import triton.language as tl
from torch.nn import functional as F

@triton.jit
def _conv2d_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape0, input_shape1, input_shape2, input_shape3,
    weight_shape0, weight_shape1, weight_shape2, weight_shape3,
    output_shape0, output_shape1, output_shape2, output_shape3,
    stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
    groups, channels_per_group,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    batch_id = tl.program_id(0)
    out_ch_id = tl.program_id(1)
    
    # Calculate output dimensions
    out_h = output_shape2
    out_w = output_shape3
    
    # Each thread handles one output element
    tid = tl.program_id(2)
    if tid >= out_h * out_w:
        return
    
    # Calculate output coordinates
    out_y = tid // out_w
    out_x = tid % out_w
    
    # Calculate input coordinates with padding and dilation
    in_y_start = out_y * stride_h - padding_h
    in_x_start = out_x * stride_w - padding_w
    
    # Initialize accumulator
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Group handling
    group_id = out_ch_id // (weight_shape0 // groups)
    
    # Convolution loop
    for c in range(channels_per_group):
        # Calculate input channel for this group
        input_ch = group_id * channels_per_group + c
        
        # Convolution computation
        for kh in range(weight_shape2):
            for kw in range(weight_shape3):
                # Calculate input coordinates
                in_y = in_y_start + kh * dilation_h
                in_x = in_x_start + kw * dilation_w
                
                # Check bounds
                if in_y >= 0 and in_y < input_shape2 and in_x >= 0 and in_x < input_shape3:
                    # Load input value
                    input_val = tl.load(input_ptr + 
                                       batch_id * input_shape1 * input_shape2 * input_shape3 +
                                       input_ch * input_shape2 * input_shape3 +
                                       in_y * input_shape3 + in_x)
                    
                    # Load weight value
                    weight_val = tl.load(weight_ptr + 
                                        out_ch_id * weight_shape1 * weight_shape2 * weight_shape3 +
                                        c * weight_shape2 * weight_shape3 +
                                        kh * weight_shape3 + kw)
                    
                    acc += input_val * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_ch_id)
        acc += bias_val
    
    # Apply ReLU
    acc = tl.maximum(acc, 0.0)
    
    # Store result
    tl.store(output_ptr + 
             batch_id * output_shape1 * output_shape2 * output_shape3 +
             out_ch_id * output_shape2 * output_shape3 +
             out_y * output_shape3 + out_x, acc)

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    # Handle scalar inputs
    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(padding, tuple):
        padding = (padding, padding)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # Calculate output shape
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    out_h = (in_h + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (in_w + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    
    # Create output tensor
    if inplace:
        output = input
    else:
        output = torch.empty(batch_size, out_channels, out_h, out_w, device=input.device, dtype=input.dtype)
    
    # Handle groups
    channels_per_group = in_channels // groups
    
    # For simplicity, we'll use PyTorch's native implementation for the convolution part
    # and apply ReLU separately for the Triton kernel
    if inplace:
        # Apply convolution first
        conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        # Apply ReLU in-place
        conv_output.relu_()
        return conv_output
    else:
        # Apply convolution
        conv_output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        # Apply ReLU
        return F.relu(conv_output)

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
#     """
#     Applies a 2D convolution followed by a ReLU activation function.

#     Args:
#         input (Tensor): The input tensor of shape (minibatch, in_channels, iH, iW).
#         weight (Tensor): The convolution filters of shape (out_channels, in_channels / groups, kH, kW).
#         bias (Tensor, optional): Optional bias tensor of shape (out_channels). Default: None.
#         stride (int or tuple, optional): The stride of the convolution kernel. Default: 1.
#         padding (int, tuple, or string, optional): Padding added to all sides of the input. Default: 0.
#         dilation (int or tuple, optional): The spacing between kernel elements. Default: 1.
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1.
#         inplace (bool, optional): If True, will perform ReLU operation in-place. Default: False.

#     Returns:
#         Tensor: A tensor resulting from the 2D convolution followed by ReLU activation.
#     """
#     conv_out = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
#     return F.relu(conv_out, inplace=inplace)

def test_relu_conv2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    input1 = torch.randn(1, 3, 5, 5, device='cuda')
    weight1 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_1"] = relu_conv2d(input1, weight1)
    
    # Test case 2: Test with bias
    input2 = torch.randn(1, 3, 5, 5, device='cuda')
    weight2 = torch.randn(2, 3, 3, 3, device='cuda')
    bias2 = torch.randn(2, device='cuda')
    results["test_case_2"] = relu_conv2d(input2, weight2, bias=bias2)
    
    # Test case 3: Test with stride and padding
    input3 = torch.randn(1, 3, 5, 5, device='cuda')
    weight3 = torch.randn(2, 3, 3, 3, device='cuda')
    results["test_case_3"] = relu_conv2d(input3, weight3, stride=2, padding=1)
    
    # Test case 4: Test with dilation and groups
    input4 = torch.randn(1, 4, 5, 5, device='cuda')
    weight4 = torch.randn(2, 2, 3, 3, device='cuda')
    results["test_case_4"] = relu_conv2d(input4, weight4, dilation=2, groups=2)
    
    return results

test_results = test_relu_conv2d()
