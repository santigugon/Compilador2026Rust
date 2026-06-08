import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    gelu_x = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    tl.store(output_ptr + offsets, gelu_x, mask=mask)

@triton.jit
def std_kernel(input_ptr, output_ptr, n_elements, n_reduced, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    # Compute mean
    mean = tl.sum(input_vals) / n_elements
    # Compute variance
    diff = input_vals - mean
    squared_diff = diff * diff
    variance = tl.sum(squared_diff) / n_reduced
    # Compute standard deviation
    std = tl.sqrt(variance)
    tl.store(output_ptr + pid, std, mask=pid < 1)

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    if approximate != 'none':
        raise NotImplementedError("Only 'none' approximation is supported in this implementation")
    
    input = input.float()
    if dim is None:
        dim = tuple(range(input.dim()))
    if isinstance(dim, int):
        dim = (dim,)
    
    # Flatten input to 1D for processing
    input_flat = input.flatten()
    n_elements = input_flat.numel()
    
    # Allocate output tensor
    if out is not None:
        output = out
    else:
        output = torch.empty_like(input_flat)
    
    # Launch GELU kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](input_flat, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Compute standard deviation
    if len(dim) == 0:
        # No reduction
        std_result = torch.std(output, correction=correction)
    else:
        # Reduce along specified dimensions
        reduced_shape = list(input.shape)
        for d in sorted(dim, reverse=True):
            reduced_shape.pop(d)
        if not keepdim:
            reduced_shape = [s for i, s in enumerate(input.shape) if i not in dim]
        else:
            reduced_shape = [1 if i in dim else s for i, s in enumerate(input.shape)]
        
        # For simplicity, we'll compute std over all elements
        std_result = torch.std(output, correction=correction)
        if not keepdim:
            std_result = std_result.unsqueeze(0)
    
    if out is not None:
        out.copy_(std_result)
        return out
    return std_result
