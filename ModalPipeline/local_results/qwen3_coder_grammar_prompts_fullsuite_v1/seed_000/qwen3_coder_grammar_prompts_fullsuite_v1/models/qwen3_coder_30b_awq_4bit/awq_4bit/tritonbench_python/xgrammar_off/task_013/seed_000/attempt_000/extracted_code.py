import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_shape, weight_shape,
    stride_h, stride_w,
    padding_h, padding_w,
    dilation_h, dilation_w,
    groups,
    BLOCK_SIZE_M=16,
    BLOCK_SIZE_N=16,
    BLOCK_SIZE_K=16
):
    # Get thread indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Load input and weight
    input_ptr = input_ptr + pid_m * BLOCK_SIZE_M * input_shape[3] + pid_k * BLOCK_SIZE_K
    weight_ptr = weight_ptr + pid_n * BLOCK_SIZE_N * weight_shape[2] * weight_shape[3] + pid_k * BLOCK_SIZE_K
    
    # Compute convolution
    output_ptr = output_ptr + pid_m * BLOCK_SIZE_M * BLOCK_SIZE_N + pid_n * BLOCK_SIZE_N
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Perform convolution computation
    for i in range(0, input_shape[2] * input_shape[3], BLOCK_SIZE_K):
        input_tile = tl.load(input_ptr + i)
        weight_tile = tl.load(weight_ptr + i)
        accumulator += tl.dot(input_tile, weight_tile)
    
    # Store result
    tl.store(output_ptr, accumulator)

@triton.jit
def batch_norm_kernel(
    input_ptr, output_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    input_shape, eps=1e-5,
    BLOCK_SIZE=256
):
    # Get thread indices
    pid = tl.program_id(0)
    
    # Load input
    input_ptr = input_ptr + pid * BLOCK_SIZE
    output_ptr = output_ptr + pid * BLOCK_SIZE
    
    # Compute mean and variance
    mean = tl.sum(input_ptr) / input_shape[0]
    var = tl.sum((input_ptr - mean) ** 2) / input_shape[0]
    
    # Normalize and scale
    normalized = (input_ptr - mean) / tl.sqrt(var + eps)
    output = normalized * weight_ptr + bias_ptr
    
    # Store result
    tl.store(output_ptr, output)

@triton.jit
def relu_kernel(
    input_ptr, output_ptr,
    input_shape,
    BLOCK_SIZE=256
):
    # Get thread indices
    pid = tl.program_id(0)
    
    # Load input
    input_ptr = input_ptr + pid * BLOCK_SIZE
    output_ptr = output_ptr + pid * BLOCK_SIZE
    
    # Apply ReLU
    output = tl.maximum(input_ptr, 0.0)
    
    # Store result
    tl.store(output_ptr, output)

@triton.jit
def dropout_kernel(
    input_ptr, output_ptr, mask_ptr,
    input_shape, p=0.5,
    BLOCK_SIZE=256
):
    # Get thread indices
    pid = tl.program_id(0)
    
    # Load input
    input_ptr = input_ptr + pid * BLOCK_SIZE
    output_ptr = output_ptr + pid * BLOCK_SIZE
    mask_ptr = mask_ptr + pid * BLOCK_SIZE
    
    # Generate random mask
    mask = tl.rand() > p
    
    # Apply dropout
    output = input_ptr * mask / (1.0 - p)
    
    # Store result
    tl.store(output_ptr, output)

def dropout_relu_batch_norm_conv2d(
    input: torch.Tensor, 
    weight: torch.Tensor, 
    bias=None, 
    stride=1, 
    padding=0, 
    dilation=1, 
    groups=1, 
    p=0.5, 
    training=True, 
    inplace=False
) -> torch.Tensor:
    # Ensure input and weight are on the same device and have correct dtype
    device = input.device
    input = input.to(torch.float32)
    weight = weight.to(torch.float32)
    
    # Conv2D
    N, C_in, H, W = input.shape
    C_out, _, kH, kW = weight.shape
    
    # Calculate output dimensions
    H_out = (H + 2 * padding - (dilation * (kH - 1) + 1)) // stride + 1
    W_out = (W + 2 * padding - (dilation * (kW - 1) + 1)) // stride + 1
    
    # Initialize output tensor
    output = torch.empty(N, C_out, H_out, W_out, device=device, dtype=torch.float32)
    
    # Conv2D kernel launch
    grid = (triton.cdiv(H_out, 16), triton.cdiv(C_out, 16), triton.cdiv(C_in, 16))
    conv2d_kernel[grid](
        input, weight, output,
        input.shape, weight.shape,
        stride, stride,
        padding, padding,
        dilation, dilation,
        groups
    )
    
    # Batch normalization
    mean = output.mean(dim=(0, 2, 3), keepdim=True)
    var = output.var(dim=(0, 2, 3), keepdim=True)
    weight_bn = torch.ones_like(mean)
    bias_bn = torch.zeros_like(mean)
    
    # Apply batch norm
    output = (output - mean) / torch.sqrt(var + 1e-5) * weight_bn + bias_bn
    
    # ReLU
    output = torch.relu(output)
    
    # Dropout
    if training and p > 0:
        mask = torch.rand_like(output) > p
        output = output * mask / (1.0 - p)
    
    return output
