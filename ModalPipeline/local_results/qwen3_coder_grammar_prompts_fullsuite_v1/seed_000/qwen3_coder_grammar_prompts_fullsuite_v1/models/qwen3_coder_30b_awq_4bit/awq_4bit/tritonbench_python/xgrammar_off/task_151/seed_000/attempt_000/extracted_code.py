import torch
import triton
import triton.language as tl

@triton.jit
def relu_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, iH, iW, oH, oW,
    kH, kW, stride, padding, dilation, groups,
    BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute output indices
    out_h = pid_m * BLOCK_SIZE_M
    out_w = pid_n * BLOCK_SIZE_N
    out_c = pid_k * BLOCK_SIZE_K
    
    # Shared memory for input tile
    input_tile = tl.shared.zeros((BLOCK_SIZE_M + 2 * padding, BLOCK_SIZE_N + 2 * padding), dtype=tl.float32)
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over groups
    for g in range(groups):
        # Compute group-specific indices
        group_in_channels = in_channels // groups
        group_out_channels = out_channels // groups
        
        # Compute group-specific output channel
        group_out_c = out_c // groups
        
        # Loop over kernel
        for kh in range(kH):
            for kw in range(kW):
                # Compute input indices
                input_h = out_h * stride + kh * dilation - padding
                input_w = out_w * stride + kw * dilation - padding
                
                # Check bounds
                if input_h >= 0 and input_h < iH and input_w >= 0 and input_w < iW:
                    # Load input tile
                    input_tile[input_h, input_w] = tl.load(input_ptr + 
                        (g * group_in_channels + 0) * iH * iW + 
                        input_h * iW + input_w)
    
    # Compute convolution
    for kh in range(kH):
        for kw in range(kW):
            # Compute weight indices
            weight_idx = (out_c // groups) * kH * kW + kh * kW + kw
            weight_val = tl.load(weight_ptr + weight_idx)
            
            # Compute convolution
            for m in range(BLOCK_SIZE_M):
                for n in range(BLOCK_SIZE_N):
                    input_val = input_tile[m + kh * dilation, n + kw * dilation]
                    acc[m, n] += input_val * weight_val
    
    # Add bias
    if bias_ptr is not None:
        for m in range(BLOCK_SIZE_M):
            for n in range(BLOCK_SIZE_N):
                acc[m, n] += tl.load(bias_ptr + out_c)
    
    # Apply ReLU
    for m in range(BLOCK_SIZE_M):
        for n in range(BLOCK_SIZE_N):
            acc[m, n] = tl.maximum(acc[m, n], 0.0)
    
    # Store output
    for m in range(BLOCK_SIZE_M):
        for n in range(BLOCK_SIZE_N):
            tl.store(output_ptr + (out_h + m) * oW + (out_w + n), acc[m, n])

def relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
    # Input validation
    if len(input.shape) != 4:
        raise ValueError("Input tensor must be 4D: (batch, channels, height, width)")
    if len(weight.shape) != 4:
        raise ValueError("Weight tensor must be 4D: (out_channels, in_channels, height, width)")
    
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Compute output dimensions
    oH = (iH + 2 * padding - (dilation * (kH - 1) + 1)) // stride + 1
    oW = (iW + 2 * padding - (dilation * (kW - 1) + 1)) // stride + 1
    
    # Initialize output tensor
    output = torch.empty(batch_size, out_channels, oH, oW, device=input.device, dtype=input.dtype)
    
    # Prepare kernel launch parameters
    grid = (
        triton.cdiv(oH, 16),
        triton.cdiv(oW, 16),
        triton.cdiv(out_channels, 16)
    )
    
    # Launch kernel
    relu_conv2d_kernel[grid](
        input,
        weight,
        bias,
        output,
        batch_size,
        in_channels,
        out_channels,
        iH,
        iW,
        oH,
        oW,
        kH,
        kW,
        stride,
        padding,
        dilation,
        groups
    )
    
    return output
