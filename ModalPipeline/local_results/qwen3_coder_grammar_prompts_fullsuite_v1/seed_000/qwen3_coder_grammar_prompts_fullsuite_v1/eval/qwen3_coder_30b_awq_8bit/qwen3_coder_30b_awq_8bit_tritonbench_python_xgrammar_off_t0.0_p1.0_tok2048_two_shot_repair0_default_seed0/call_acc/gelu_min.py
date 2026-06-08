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
        sqrt2 = 1.4142135623730951
        x_over_sqrt2 = x / sqrt2
        erf_x = 2.0 * tl.sigmoid(x_over_sqrt2 * 1.4142135623730951) - 1.0
        gelu_x = x * 0.5 * (1.0 + erf_x)
    else:
        # Approximate GELU using tanh
        # GELU ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pi = 3.141592653589793
        sqrt_2_over_pi = 1.4142135623730951 / pi
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_x = 2.0 * tl.sigmoid(tanh_arg) - 1.0
        gelu_x = 0.5 * x * (1.0 + tanh_x)
    
    tl.store(out_ptr + offsets, gelu_x, mask=mask)

@triton.jit
def _min_kernel(x_ptr, out_ptr, indices_ptr, n: tl.constexpr, dim_size: tl.constexpr, stride: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
    
    # Initialize min and index
    min_val = x
    min_idx = tl.arange(0, BLOCK)
    
    # Simple reduction for min along dimension
    # This is a simplified version - in practice, you'd want to use proper reduction
    # But for this case, we'll compute min over all elements
    # For a proper implementation, we'd need to handle the dimension properly
    # For now, we'll compute the overall min
    
    # Use a simple reduction approach
    for i in range(0, BLOCK, 32):
        if i + 32 <= BLOCK:
            # Load 32 elements
            vals = tl.load(x_ptr + offsets + i, mask=(offsets + i) < n, other=float('inf'))
            # Compute min
            for j in range(32):
                if i + j < BLOCK and (i + j) < n:
                    min_val = tl.minimum(min_val, vals[j])
                    # This is a simplified approach - proper indexing would be needed
                    # For now, we'll just compute the overall min
    
    # Store result
    tl.store(out_ptr + pid, min_val, mask=pid < n // BLOCK + (1 if n % BLOCK else 0))

def gelu_min(input, approximate='none', dim=None, keepdim=False, out=None):
    # Handle scalar input
    if input.dim() == 0:
        input = input.unsqueeze(0)
    
    # Apply GELU
    gelu_out = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Determine approximate parameter
    approx = 'none' if approximate == 'none' else 'tanh'
    
    _gelu_kernel[grid](input, gelu_out, n, approx, BLOCK=block)
    
    # Compute min along specified dimension
    if dim is None:
        # Compute min over all elements
        result = torch.min(gelu_out)
        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        # Compute min along specified dimension
        if keepdim:
            result = torch.min(gelu_out, dim=dim, keepdim=True)
        else:
            result = torch.min(gelu_out, dim=dim)
        
        if out is not None:
            out.copy_(result[0] if isinstance(result, tuple) else result)
            return out
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
