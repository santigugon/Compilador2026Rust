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
    
    # Compute the output tensor indices
    out_idx = (m // BLOCK_SIZE_M) * BLOCK_SIZE_M + (n // BLOCK_SIZE_N) * BLOCK_SIZE_N
    
    # Compute the input tensor indices
    input_idx = (m // BLOCK_SIZE_M) * BLOCK_SIZE_M + (n // BLOCK_SIZE_N) * BLOCK_SIZE_N
    
    # Compute the weight tensor indices
    weight_idx = (m // BLOCK_SIZE_M) * BLOCK_SIZE_M + (n // BLOCK_SIZE_N) * BLOCK_SIZE_N
    
    # Compute the bias tensor indices
    bias_idx = (m // BLOCK_SIZE_M) * BLOCK_SIZE_M + (n // BLOCK_SIZE_N) * BLOCK_SIZE_N
    
    # Compute the output tensor
    output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute the convolution
    for i in range(0, kH * kW):
        # Compute the input tensor indices
        input_h = (m // BLOCK_SIZE_M) * stride_h + (i // kW) * dilation_h - pad_h
        input_w = (n // BLOCK_SIZE_N) * stride_w + (i % kW) * dilation_w - pad_w
        
        # Check if the input tensor indices are valid
        if input_h >= 0 and input_h < iH and input_w >= 0 and input_w < iW:
            # Compute the weight tensor indices
            weight_h = (i // kW) * dilation_h
            weight_w = (i % kW) * dilation_w
            
            # Compute the output tensor
            output += tl.load(input_ptr + input_h * iW + input_w) * tl.load(weight_ptr + weight_h * kW + weight_w)
    
    # Apply the bias
    if bias_ptr is not None:
        output += tl.load(bias_ptr + (m // BLOCK_SIZE_M) * BLOCK_SIZE_M + (n // BLOCK_SIZE_N) * BLOCK_SIZE_N)
    
    # Apply the sigmoid activation function
    output = tl.sigmoid(output)
    
    # Store the output tensor
    tl.store(output_ptr + out_idx, output)

def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None):
    # Get the input tensor dimensions
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Compute the output tensor dimensions
    oH = (iH + 2 * padding - (dilation * (kH - 1) + 1)) // stride + 1
    oW = (iW + 2 * padding - (dilation * (kW - 1) + 1)) // stride + 1
    
    # Create the output tensor if not provided
    if out is None:
        out = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Create the bias tensor if not provided
    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)
    
    # Create the kernel
    kernel = conv2d_sigmoid_kernel
    
    # Launch the kernel
    grid = (triton.cdiv(oH, 16), triton.cdiv(oW, 16), 1)
    kernel(
        input, weight, bias, out,
        batch_size, in_channels, out_channels, iH, iW, oH, oW,
        kH, kW, stride, stride, padding, padding, dilation, dilation,
        groups, in_channels // groups,
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16
    )
    
    return out
