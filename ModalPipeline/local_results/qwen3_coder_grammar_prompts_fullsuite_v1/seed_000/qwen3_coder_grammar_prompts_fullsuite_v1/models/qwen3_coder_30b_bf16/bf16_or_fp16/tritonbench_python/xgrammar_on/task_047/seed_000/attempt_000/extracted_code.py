import torch
import triton
import triton.language as tl

@triton.jit
def selu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Applies SELU activation function: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))"""
    # Compute the block index
    block_idx = tl.program_id(0)
    # Compute the starting element index for this block
    start_idx = block_idx * BLOCK_SIZE
    # Create a mask for valid elements
    mask = start_idx + tl.arange(0, BLOCK_SIZE) < n_elements
    # Load data
    x = tl.load(x_ptr + start_idx + tl.arange(0, BLOCK_SIZE), mask=mask)
    # SELU constants
    alpha = 1.6733
    scale = 1.0507
    # Compute SELU: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    exp_x = tl.exp(x)
    selu_x = scale * (tl.maximum(0, x) + tl.minimum(0, alpha * (exp_x - 1.0)))
    # Store result
    tl.store(output_ptr + start_idx + tl.arange(0, BLOCK_SIZE), selu_x, mask=mask)

def selu(input, inplace=False):
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Get the number of elements
    n_elements = input.numel()
    
    # Launch the kernel
    grid = (triton.cdiv(n_elements, 1024),)
    selu_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)
    
    return output