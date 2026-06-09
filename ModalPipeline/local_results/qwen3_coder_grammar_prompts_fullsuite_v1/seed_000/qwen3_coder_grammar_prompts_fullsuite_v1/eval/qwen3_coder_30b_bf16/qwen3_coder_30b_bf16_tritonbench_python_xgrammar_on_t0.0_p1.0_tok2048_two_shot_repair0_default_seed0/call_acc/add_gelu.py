import torch
import triton
import triton.language as tl

def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr, approximate: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation
    if approximate == 'tanh':
        # GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
        x_cubed = x * x * x
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        gelu_x = 0.5 * x * (1.0 + tl.tanh(tanh_arg))
    else:
        # Standard GELU: x * Phi(x) where Phi is the standard normal CDF
        # Approximation: x * 0.5 * (1 + tanh(sqrt(0.5) * (x + 0.044715 * x^3)))
        sqrt_05 = 0.7071067811865476  # sqrt(0.5)
        x_cubed = x * x * x
        tanh_arg = sqrt_05 * (x + 0.044715 * x_cubed)
        gelu_x = x * 0.5 * (1.0 + tl.tanh(tanh_arg))
    
    tl.store(out_ptr + offsets, gelu_x, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Broadcast tensors
    input, other = torch.broadcast_tensors(input, other)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError(f"Output tensor shape {out.shape} does not match input shape {input.shape}")
        if out.dtype != input.dtype:
            raise ValueError(f"Output tensor dtype {out.dtype} does not match input dtype {input.dtype}")
    
    # Compute input + alpha * other
    temp = input + alpha * other
    
    # Apply GELU
    n = temp.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _gelu_kernel[grid](temp, out, n, BLOCK=block, approximate=approximate)
    return out
##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_add_gelu():
    results = {}

    # Test case 1: Basic test with default parameters
    input_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    other_tensor = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    results["test_case_1"] = add_gelu(input_tensor, other_tensor)

    # Test case 2: Test with alpha parameter
    alpha = 2
    results["test_case_2"] = add_gelu(input_tensor, other_tensor, alpha=alpha)

    # Test case 3: Test with approximate='tanh'
    approximate = 'tanh'
    results["test_case_3"] = add_gelu(input_tensor, other_tensor, approximate=approximate)

    # Test case 4: Test with a scalar 'other'
    other_scalar = 0.5
    results["test_case_4"] = add_gelu(input_tensor, other_scalar)

    return results

test_results = test_add_gelu()
