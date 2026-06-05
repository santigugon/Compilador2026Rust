import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    APPROXIMATE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    
    if APPROXIMATE == "tanh":
        # Approximate GELU using tanh
        output = 0.5 * input * (1 + tl.tanh(1.702 * input))
    else:
        # Exact GELU using error function
        output = 0.5 * input * (1 + tl.erf(input / tl.sqrt(2.0)))
    
    tl.store(output_ptr + offsets, output, mask=mask)

def gelu(input, approximate='none'):
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function element-wise to the input tensor.
    
    Args:
        input (torch.Tensor): Input tensor
        approximate (str): Approximation method, either 'none' for exact GELU or 'tanh' for approximate GELU
    
    Returns:
        torch.Tensor: Output tensor with GELU applied
    """
    output = torch.empty_like(input)
    
    # Determine the approximate method
    approximate_method = "tanh" if approximate == "tanh" else "none"
    
    # Calculate grid and block size
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    gelu_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        APPROXIMATE=approximate_method
    )
    
    return output
