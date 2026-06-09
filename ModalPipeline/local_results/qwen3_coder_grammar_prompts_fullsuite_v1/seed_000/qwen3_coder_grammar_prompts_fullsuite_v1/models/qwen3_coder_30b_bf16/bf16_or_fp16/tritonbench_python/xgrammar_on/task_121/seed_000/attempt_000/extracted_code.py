import torch
import triton
import triton.language as tl

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

def gelu_tanh(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # GELU approximation using tanh
    y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _gelu_exact_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Exact GELU
    y = 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _min_kernel(x_ptr, out_ptr, size: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr):
    # This kernel computes the minimum along a single dimension
    # For simplicity, we'll compute the minimum over all elements
    # In a real implementation, this would need to be more complex
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
    # Compute block minimum
    block_min = tl.min(x, axis=0)
    # Store the minimum
    tl.store(out_ptr, block_min)

@triton.jit
def _min_kernel_dim(x_ptr, out_ptr, size: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr):
    # This kernel computes the minimum along a specific dimension
    # For simplicity, we'll compute the minimum over all elements
    # In a real implementation, this would need to be more complex
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    x = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
    # Compute block minimum
    block_min = tl.min(x, axis=0)
    # Store the minimum
    tl.store(out_ptr, block_min)

def min_gelu(input, dim=None, keepdim=False, approximate='none', out=None):
    if out is not None:
        # If out is provided, we need to handle it properly
        # For now, we'll just compute the result and copy to out
        result = min_gelu(input, dim, keepdim, approximate)
        out.copy_(result)
        return out
    
    # Apply GELU
    if approximate == 'none':
        # Use exact GELU
        gelu_input = input * 0.5 * (1.0 + torch.erf(input / torch.sqrt(torch.tensor(2.0))))
    elif approximate == 'tanh':
        # Use tanh approximation
        gelu_input = 0.5 * input * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (input + 0.044715 * torch.pow(input, 3))))
    else:
        raise ValueError(f"Unsupported approximate method: {approximate}")
    
    # Compute minimum
    if dim is None:
        # Return minimum over all elements
        return torch.min(gelu_input)
    else:
        # Return minimum along specified dimension
        return torch.min(gelu_input, dim=dim, keepdim=keepdim)[0]