import torch
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 'none':
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        x_over_sqrt2 = x / sqrt_2
        erf_x = 2.0 * tl.sigmoid(0.5 * x_over_sqrt2 * x_over_sqrt2) - 1.0
        gelu_x = x * 0.5 * (1.0 + erf_x)
    else:
        # Approximate GELU using tanh
        # GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pi = 3.141592653589793
        sqrt_2_over_pi = 1.4142135623730951 / pi
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_x = 2.0 * tl.sigmoid(2.0 * tanh_arg) - 1.0
        gelu_x = 0.5 * x * (1.0 + tanh_x)
    
    tl.store(out_ptr + offsets, gelu_x, mask=mask)

@triton.jit
def _min_kernel(x_ptr, out_ptr, indices_ptr, n: tl.constexpr, dim_size: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # For simplicity, we compute min along the last dimension
    # In a full implementation, we would need to handle arbitrary dimensions
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
    
    # Simple reduction to find minimum
    min_val = tl.min(x)
    min_idx = tl.argmin(x)
    
    tl.store(out_ptr + pid, min_val, mask=pid < dim_size)
    tl.store(indices_ptr + pid, min_idx, mask=pid < dim_size)

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    # Handle scalar input
    if input.dim() == 0:
        input = input.unsqueeze(0)
        squeeze_dim = True
    else:
        squeeze_dim = False
    
    # Apply GELU
    gelu_input = input.float()
    out_gelu = torch.empty_like(gelu_input)
    
    n = gelu_input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    approximate_val = 'none' if approximate == 'none' else 'tanh'
    _gelu_kernel[grid](gelu_input, out_gelu, n, approximate_val, BLOCK=block)
    
    # Convert back to original dtype
    out_gelu = out_gelu.to(input.dtype)
    
    # Handle min computation
    if dim is None:
        # Compute min over all elements
        result = torch.min(out_gelu)
        indices = torch.tensor([0], dtype=torch.long)  # dummy index
    else:
        # Compute min along specified dimension
        # For simplicity, we'll compute min along the last dimension
        # In a full implementation, we'd need to handle arbitrary dimensions properly
        if dim < 0:
            dim = out_gelu.dim() + dim
            
        if dim >= out_gelu.dim():
            raise ValueError("dim out of range")
            
        # For this implementation, we'll compute min along the last dimension
        # This is a simplified version - a full implementation would be more complex
        if dim == out_gelu.dim() - 1:
            # Simple case: min along last dimension
            result = torch.min(out_gelu, dim=dim, keepdim=keepdim)
            if keepdim:
                result = result.values
            else:
                result = result.values
        else:
            # For other dimensions, we fall back to PyTorch
            result = torch.min(out_gelu, dim=dim, keepdim=keepdim)
            if keepdim:
                result = result.values
            else:
                result = result.values
    
    # Handle output tensor
    if out is not None:
        out.copy_(result)
        return out
    
    if squeeze_dim:
        result = result.squeeze(0)
    
    return result

##################################################################################################################################################



def test_gelu_min():
    results = {}

    # Test case 1: Default approximate='none', no dim, no keepdim
    input_tensor = torch.tensor([0.5, -0.5, 1.0, -1.0], device='cuda')
    results['test_case_1'] = gelu_min(input_tensor)

    # Test case 2: approximate='tanh', no dim, no keepdim
    input_tensor = torch.tensor([0.5, -0.5, 1.0, -1.0], device='cuda')
    results['test_case_2'] = gelu_min(input_tensor, approximate='tanh')

    # Test case 3: approximate='none', with dim, no keepdim
    input_tensor = torch.tensor([[0.5, -0.5], [1.0, -1.0]], device='cuda')
    results['test_case_3'] = gelu_min(input_tensor, dim=1)

    # Test case 4: approximate='tanh', with dim, keepdim=True
    input_tensor = torch.tensor([[0.5, -0.5], [1.0, -1.0]], device='cuda')
    results['test_case_4'] = gelu_min(input_tensor, approximate='tanh', dim=1, keepdim=True)

    return results

test_results = test_gelu_min()
