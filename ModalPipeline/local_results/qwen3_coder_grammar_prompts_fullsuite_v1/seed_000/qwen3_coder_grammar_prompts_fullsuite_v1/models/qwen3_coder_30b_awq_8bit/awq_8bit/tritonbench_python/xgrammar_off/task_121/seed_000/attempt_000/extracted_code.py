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
    # GELU approximation using tanh
    y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def min_kernel(x_ptr, y_ptr, n_elements, dim_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute minimum along the specified dimension
    min_val = tl.min(x)
    tl.store(y_ptr + pid, min_val, mask=pid < dim_size)

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    if approximate == 'none':
        # Use exact GELU computation
        input = torch.nn.functional.gelu(input)
    elif approximate == 'tanh':
        # Use tanh approximation
        input = 0.5 * input * (1.0 + torch.tanh(0.7978845608028654 * input * (1.0 + 0.044715 * input * input)))
    else:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    if dim is None:
        # Reduce all elements
        return torch.min(input)
    else:
        # Reduce along specified dimension
        return torch.min(input, dim=dim, keepdim=keepdim)[0]
