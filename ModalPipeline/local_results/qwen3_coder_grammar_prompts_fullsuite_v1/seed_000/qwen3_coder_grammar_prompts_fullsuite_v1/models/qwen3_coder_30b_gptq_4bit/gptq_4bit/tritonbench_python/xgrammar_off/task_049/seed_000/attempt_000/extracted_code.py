import torch
import triton
import triton.language as tl
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_row, input_col, output_row, output_col,
    weight_row, weight_col,
    stride, padding, dilation,
    groups, negative_slope,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INPLACE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output indices
    output_m = pid_m * BLOCK_M
    output_n = pid_n * BLOCK_N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the kernel
    for k in range(0, K, BLOCK_K):
        # Load input and weight tiles
        input_tile = tl.load(
            input_ptr + (output_m + tl.arange(0, BLOCK_M)[:, None]) * input_col + 
            (k + tl.arange(0, BLOCK_K)[None, :]) * input_row + 
            (output_n + tl.arange(0, BLOCK_N)[:, None]) * input_col + 
            (k + tl.arange(0, BLOCK_K)[None, :]) * input_row,
            mask=(output_m + tl.arange(0, BLOCK_M)[:, None] < output_row) &
                  (output_n + tl.arange(0, BLOCK_N)[:, None] < output_col),
            other=0.0
        )
        
        weight_tile = tl.load(
            weight_ptr + (k + tl.arange(0, BLOCK_K)[None, :]) * weight_col + 
            (output_m + tl.arange(0, BLOCK_M)[:, None]) * weight_row + 
            (output_n + tl.arange(0, BLOCK_N)[:, None]) * weight_row,
            mask=(k + tl.arange(0, BLOCK_K)[None, :] < K) &
                  (output_m + tl.arange(0, BLOCK_M)[:, None] < M) &
                  (output_n + tl.arange(0, BLOCK_N)[:, None] < N),
            other=0.0
        )
        
        # Accumulate
        acc += tl.dot(input_tile, weight_tile)
    
    # Apply bias if present
    if bias_ptr is not None:
        bias_tile = tl.load(bias_ptr + tl.arange(0, BLOCK_N), mask=tl.arange(0, BLOCK_N) < N)
        acc += bias_tile[None, :]
    
    # Apply Leaky ReLU
    acc = tl.where(acc < 0, acc * negative_slope, acc)
    
    # Store result
    tl.store(
        output_ptr + (output_m + tl.arange(0, BLOCK_M)[:, None]) * output_col + 
        (output_n + tl.arange(0, BLOCK_N)[:, None]),
        acc,
        mask=(output_m + tl.arange(0, BLOCK_M)[:, None] < output_row) &
              (output_n + tl.arange(0, BLOCK_N)[:, None] < output_col)
    )

def leaky_relu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, negative_slope=0.01, inplace=False):
    # Validate input dimensions
    assert input.dim() == 4, "Input must be a 4D tensor (N, C, H, W)"
    assert weight.dim() == 4, "Weight must be a 4D tensor (out_channels, in_channels, kernel_h, kernel_w)"
    
    # Get dimensions
    batch_size, in_channels, input_h, input_w = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Calculate output dimensions
    output_h = (input_h + 2 * padding - (dilation * (kernel_h - 1) + 1)) // stride + 1
    output_w = (input_w + 2 * padding - (dilation * (kernel_w - 1) + 1)) // stride + 1
    
    # Reshape input for convolution
    input_reshaped = input.view(batch_size, in_channels, input_h, input_w)
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, output_h, output_w, device=input.device, dtype=input.dtype)
    
    # Set up grid
    grid = (triton.cdiv(output_h, 16), triton.cdiv(output_w, 16))
    
    # Launch kernel
    conv2d_kernel[grid](
        input_reshaped,
        weight,
        bias,
        output,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride,
        padding,
        dilation,
        groups,
        negative_slope,
        output_h * output_w,
        output_h * output_w,
        in_channels * kernel_h * kernel_w,
        INPLACE=False
    )
    
    return output
