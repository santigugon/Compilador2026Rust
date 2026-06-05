import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_shape_0, input_shape_1, input_shape_2, input_shape_3,
    weight_shape_0, weight_shape_1, weight_shape_2, weight_shape_3,
    output_shape_0, output_shape_1, output_shape_2, output_shape_3,
    stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(output_shape_2, BLOCK_SIZE_M)
    num_pid_m = tl.cdiv(output_shape_3, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    output_offset = pid_m * output_shape_3 + pid_n
    output_ptr += output_offset
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, weight_shape_1 * weight_shape_2 * weight_shape_3, BLOCK_SIZE_K):
        input_offset = k * input_shape_1 * input_shape_2 * input_shape_3
        weight_offset = k * weight_shape_1 * weight_shape_2 * weight_shape_3
        input_ptr += input_offset
        weight_ptr += weight_offset
        
        input_tile = tl.load(input_ptr + tl.arange(0, BLOCK_SIZE_M)[:, None] * input_shape_3 + tl.arange(0, BLOCK_SIZE_K)[None, :])
        weight_tile = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE_K)[:, None] * weight_shape_3 + tl.arange(0, BLOCK_SIZE_N)[None, :])
        
        acc += tl.dot(input_tile, weight_tile)
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE_M))
        acc += bias[:, None]
    
    output = tl.sigmoid(acc)
    tl.store(output_ptr, output)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    oH = (iH + 2 * pad_h - (dilation_h * (kH - 1) + 1)) // stride_h + 1
    oW = (iW + 2 * pad_w - (dilation_w * (kW - 1) + 1)) // stride_w + 1
    
    if out is None:
        out = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=torch.float32)
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    num_warps = 4
    num_stages = 3
    
    grid = (triton.cdiv(oH, BLOCK_SIZE_M) * triton.cdiv(oW, BLOCK_SIZE_N),)
    
    sigmoid_conv2d_kernel[grid](
        input, weight, bias, out,
        batch_size, in_channels, iH, iW,
        out_channels, in_channels // groups, kH, kW,
        batch_size, out_channels, oH, oW,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        num_warps=num_warps, num_stages=num_stages
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
