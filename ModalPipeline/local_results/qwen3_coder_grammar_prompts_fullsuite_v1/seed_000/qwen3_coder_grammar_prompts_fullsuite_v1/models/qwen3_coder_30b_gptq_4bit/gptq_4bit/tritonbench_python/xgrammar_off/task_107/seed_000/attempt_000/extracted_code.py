import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU
    if approximate == 'none':
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        erf_x = tl.erf(x / sqrt_2)
        gelu_x = x * 0.5 * (1.0 + erf_x)
    else:
        # Approximate GELU using tanh
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pi = 3.14159265358979323846
        sqrt_2_over_pi = 1.4142135623730951 / 3.14159265358979323846
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_x = tl.tanh(tanh_arg)
        gelu_x = 0.5 * x * (1.0 + tanh_x)
    
    tl.store(out_ptr + offsets, gelu_x, mask=mask)

@triton.jit
def _min_kernel(x_ptr, out_ptr, indices_ptr, n: tl.constexpr, dim: tl.constexpr, keepdim: tl.constexpr, BLOCK: tl.constexpr):
    # This is a simplified version for demonstration
    # In practice, a more complex reduction kernel would be needed
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For simplicity, we'll just return the first element
    # A full implementation would require proper reduction logic
    if dim == -1:
        # Reduce over all elements
        if pid == 0:
            min_val = x[0]
            min_idx = 0
            for i in range(1, min(BLOCK, n)):
                if x[i] < min_val:
                    min_val = x[i]
                    min_idx = i
            tl.store(out_ptr, min_val)
            tl.store(indices_ptr, min_idx)
    else:
        # For this implementation, we'll just return the first element
        tl.store(out_ptr + offsets, x, mask=mask)

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    # Convert approximate to string if needed
    if approximate not in ['none', 'tanh']:
        raise ValueError("approximate must be 'none' or 'tanh'")
    
    # Apply GELU
    input_gelu = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _gelu_kernel[grid](input, input_gelu, n, approximate, BLOCK=block)
    
    # Compute minimum
    if dim is None:
        # Reduce over all elements
        result = input_gelu.min(keepdim=keepdim)
        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        # Reduce along specified dimension
        result = input_gelu.min(dim=dim, keepdim=keepdim)
        if out is not None:
            out.copy_(result)
            return out
        return result
