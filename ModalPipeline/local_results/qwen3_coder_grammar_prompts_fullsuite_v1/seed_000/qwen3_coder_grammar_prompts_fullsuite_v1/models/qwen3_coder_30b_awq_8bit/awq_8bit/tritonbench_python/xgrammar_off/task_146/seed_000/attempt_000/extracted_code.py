import torch
import triton
import triton.language as tl

@triton.jit
def elu_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_size, weight_size, output_size,
    alpha, inplace,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size
    
    # Load input and weight
    input_block = tl.load(input_ptr + offsets, mask=mask)
    weight_block = tl.load(weight_ptr + offsets, mask=mask)
    
    # Linear transformation
    linear_result = input_block * weight_block
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_block = tl.load(bias_ptr + offsets, mask=mask)
        linear_result += bias_block
    
    # Apply ELU
    elu_result = tl.where(linear_result > 0, linear_result, alpha * (tl.exp(linear_result) - 1))
    
    # Store result
    if inplace:
        tl.store(input_ptr + offsets, elu_result, mask=mask)
    else:
        tl.store(output_ptr + offsets, elu_result, mask=mask)

def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    # Ensure input and weight are contiguous
    input = input.contiguous()
    weight = weight.contiguous()
    
    # Check dimensions
    if input.dim() != 1 or weight.dim() != 1:
        raise ValueError("Input and weight must be 1D tensors")
    
    if input.size(0) != weight.size(0):
        raise ValueError("Input size must match weight size")
    
    if bias is not None and bias.size(0) != weight.size(0):
        raise ValueError("Bias size must match weight size")
    
    # Prepare output tensor
    if not inplace:
        output = torch.empty_like(input)
    else:
        output = input
    
    # Get sizes
    input_size = input.numel()
    weight_size = weight.numel()
    output_size = input_size
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(output_size, BLOCK_SIZE),)
    
    # Determine bias pointer
    bias_ptr = bias.data_ptr() if bias is not None else None
    
    # Launch kernel
    elu_linear_kernel[grid](
        input.data_ptr(),
        weight.data_ptr(),
        bias_ptr,
        output.data_ptr(),
        input_size,
        weight_size,
        output_size,
        alpha,
        inplace,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
