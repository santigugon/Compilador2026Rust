import torch
import triton
import triton.language as tl

@triton.jit
def pixel_shuffle_conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    in_channels, out_channels, iH, iW, oH, oW,
    kH, kW, stride, padding, dilation, groups, upscale_factor,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Get block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Compute output indices
    out_h = pid_m * BLOCK_SIZE_M
    out_w = pid_n * BLOCK_SIZE_N
    group_id = pid_k
    
    # Shared memory for input tiles
    input_tile = tl.shared_ptr(input_ptr, (BLOCK_SIZE_M, BLOCK_SIZE_K), (1, 1))
    weight_tile = tl.shared_ptr(weight_ptr, (BLOCK_SIZE_K, BLOCK_SIZE_N), (1, 1))
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Perform convolution
    for k in range(0, in_channels, BLOCK_SIZE_K):
        # Load input tile
        input_tile = tl.load(input_ptr + (out_h * iW + out_w) * in_channels + k)
        # Load weight tile
        weight_tile = tl.load(weight_ptr + (k * out_channels + out_w) * kH * kW)
        # Compute dot product
        acc += tl.dot(input_tile, weight_tile)
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_tile = tl.load(bias_ptr + out_w)
        acc += bias_tile
    
    # Apply pixel shuffle
    shuffle_h = out_h // upscale_factor
    shuffle_w = out_w // upscale_factor
    shuffle_c = out_w % upscale_factor
    
    # Store result
    tl.store(output_ptr + (shuffle_h * oW + shuffle_w) * out_channels + shuffle_c, acc)

def pixel_shuffle_conv2d(input: torch.Tensor, weight: torch.Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1, upscale_factor=2) -> torch.Tensor:
    # Input shapes
    batch_size, in_channels, iH, iW = input.shape
    out_channels, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    oH = (iH + 2 * padding - (dilation * (kH - 1) + 1)) // stride + 1
    oW = (iW + 2 * padding - (dilation * (kW - 1) + 1)) // stride + 1
    
    # Calculate output shape after pixel shuffle
    out_H = oH * upscale_factor
    out_W = oW * upscale_factor
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, out_H, out_W, device=input.device, dtype=input.dtype)
    
    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Launch kernel
    grid = (
        triton.cdiv(out_H, BLOCK_SIZE_M),
        triton.cdiv(out_W, BLOCK_SIZE_N),
        groups
    )
    
    # Create kernel
    kernel = pixel_shuffle_conv2d_kernel
    
    # Launch kernel with appropriate parameters
    kernel[grid](
        input,
        weight,
        bias,
        output,
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
        groups,
        upscale_factor,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output
