import torch
import triton
import triton.language as tl

@triton.jit
def _fused_conv2d_selu_norm_kernel(
    input_ptr,  # pointer to input tensor
    weight_ptr,  # pointer to weight tensor
    bias_ptr,  # pointer to bias tensor
    output_ptr,  # pointer to output tensor
    mean_ptr,  # pointer to mean tensor
    var_ptr,  # pointer to variance tensor
    weight_norm_ptr,  # pointer to weight for normalization
    bias_norm_ptr,  # pointer to bias for normalization
    batch_size,  # batch size
    in_channels,  # input channels
    out_channels,  # output channels
    iH, iW,  # input height and width
    oH, oW,  # output height and width
    kH, kW,  # kernel height and width
    stride_h, stride_w,  # stride height and width
    padding_h, padding_w,  # padding height and width
    dilation_h, dilation_w,  # dilation height and width
    groups,  # number of groups
    eps,  # epsilon for normalization
    momentum,  # momentum for running stats
    affine,  # whether to use affine parameters
    track_running_stats,  # whether to track running stats
    BLOCK_SIZE_M,  # block size for M dimension
    BLOCK_SIZE_N,  # block size for N dimension
    BLOCK_SIZE_K,  # block size for K dimension
):
    # Get the thread index
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute the output index
    output_idx = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    input_idx = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    weight_idx = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    # Load input, weight, and bias
    input_block = tl.load(input_ptr + output_idx[:, None, None] * iH * iW + input_idx[None, :, None] * iW + weight_idx[None, None, :])
    weight_block = tl.load(weight_ptr + output_idx[:, None, None] * in_channels * kH * kW + input_idx[None, :, None] * kH * kW + weight_idx[None, None, :])
    bias_block = tl.load(bias_ptr + output_idx)
    
    # Perform convolution
    conv_result = tl.sum(input_block * weight_block, axis=2)
    conv_result += bias_block
    
    # Apply SELU activation
    selu_result = tl.where(conv_result > 0, conv_result, 1.0507009873554804934193349852946 * tl.exp(conv_result) - 1.0507009873554804934193349852946)
    
    # Apply instance normalization
    mean = tl.sum(selu_result, axis=1) / (oH * oW)
    var = tl.sum((selu_result - mean[:, None]) ** 2, axis=1) / (oH * oW)
    
    # Update running stats if needed
    if track_running_stats:
        mean = (1 - momentum) * mean + momentum * mean_ptr
        var = (1 - momentum) * var + momentum * var_ptr
        
    # Normalize
    norm_result = (selu_result - mean[:, None]) / tl.sqrt(var[:, None] + eps)
    
    # Apply affine transformation if needed
    if affine:
        norm_result = weight_norm_ptr * norm_result + bias_norm_ptr
    
    # Store result
    tl.store(output_ptr + output_idx[:, None] * oH * oW + input_idx[None, :] * oW + weight_idx[None, None, :], norm_result)


def fused_instance_norm_selu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_features=None, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
    # Ensure input is contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    
    # Get dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Compute output dimensions
    oH = (iH + 2 * padding - (dilation * (kH - 1) + 1)) // stride + 1
    oW = (iW + 2 * padding - (dilation * (kW - 1) + 1)) // stride + 1
    
    # Initialize output tensor
    output = torch.empty((batch_size, out_channels, oH, oW), device=input.device, dtype=input.dtype)
    
    # Initialize mean and variance tensors
    mean = torch.empty((out_channels,), device=input.device, dtype=torch.float32)
    var = torch.empty((out_channels,), device=input.device, dtype=torch.float32)
    
    # Initialize affine parameters if needed
    weight_norm = torch.ones((out_channels,), device=input.device, dtype=torch.float32) if affine else None
    bias_norm = torch.zeros((out_channels,), device=input.device, dtype=torch.float32) if affine else None
    
    # Define block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Launch kernel
    grid = (
        triton.cdiv(out_channels, BLOCK_SIZE_M),
        triton.cdiv(iH * iW, BLOCK_SIZE_N),
        triton.cdiv(in_channels * kH * kW, BLOCK_SIZE_K)
    )
    
    # Call kernel
    _fused_conv2d_selu_norm_kernel[grid](
        input, weight, bias, output, mean, var, weight_norm, bias_norm,
        batch_size, in_channels, out_channels, iH, iW, oH, oW,
        kH, kW, stride, stride, padding, padding, dilation, dilation,
        groups, eps, momentum, affine, track_running_stats,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output