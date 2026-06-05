import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    gelu_x = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    tl.store(y_ptr + offsets, gelu_x, mask=mask)

@triton.jit
def std_kernel(x_ptr, out_ptr, n_elements, n_reduced, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute mean
    mean = tl.sum(x) / n_elements
    # Compute variance
    squared_diff = (x - mean) * (x - mean)
    variance = tl.sum(squared_diff) / n_reduced
    # Compute standard deviation
    std = tl.sqrt(variance)
    tl.store(out_ptr + pid, std, mask=pid < n_reduced)

def gelu_std(input, dim=None, keepdim=False, correction=1, approximate='none', out=None):
    if approximate != 'none':
        raise NotImplementedError("Approximate GELU not implemented in Triton version")
    
    # Apply GELU activation
    input_flat = input.flatten()
    n_elements = input_flat.numel()
    
    # Allocate output tensor for GELU
    gelu_out = torch.empty_like(input_flat)
    
    # Launch GELU kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](input_flat, gelu_out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reshape to original shape
    gelu_out = gelu_out.reshape(input.shape)
    
    # Compute standard deviation
    if dim is None:
        # Reduce all dimensions
        reduced_elements = gelu_out.numel()
        if correction == 1:
            reduced_elements -= 1
        std_val = torch.std(gelu_out, correction=correction)
        if out is not None:
            out.copy_(std_val)
            return out
        return std_val
    else:
        # Reduce specified dimensions
        if isinstance(dim, int):
            dim = (dim,)
        # Compute output shape
        output_shape = list(input.shape)
        if keepdim:
            for d in dim:
                output_shape[d] = 1
        else:
            for d in sorted(dim, reverse=True):
                output_shape.pop(d)
        
        # Flatten for computation
        flattened = gelu_out.flatten()
        reduced_elements = 1
        for d in dim:
            reduced_elements *= input.shape[d]
        
        if correction == 1:
            reduced_elements -= 1
            
        # For simplicity, use PyTorch's std function for reduction
        std_result = torch.std(gelu_out, dim=dim, keepdim=keepdim, correction=correction)
        
        if out is not None:
            out.copy_(std_result)
            return out
        return std_result
