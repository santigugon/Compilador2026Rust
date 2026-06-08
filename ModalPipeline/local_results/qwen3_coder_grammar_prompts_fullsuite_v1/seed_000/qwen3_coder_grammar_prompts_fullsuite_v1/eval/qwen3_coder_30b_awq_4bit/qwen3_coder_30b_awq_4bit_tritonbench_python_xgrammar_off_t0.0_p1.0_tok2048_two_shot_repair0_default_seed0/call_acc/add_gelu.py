import torch
import triton
import triton.language as tl

@triton.jit
def _add_gelu_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Compute x + alpha * y
    sum_val = x + alpha * y
    # Apply GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using the standard approximation for better numerical stability
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (sum_val + 0.044715 * x_cubed)
    tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * tanh_arg)) - 1.0
    gelu_val = 0.5 * sum_val * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None):
    # Handle scalar other case
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the same device and dtype as input
    other = other.to(input.device, input.dtype)
    
    # Handle broadcasting
    if input.shape != other.shape:
        # Use torch's broadcasting rules
        input, other = torch.broadcast_tensors(input, other)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        if out.shape != input.shape:
            raise ValueError("Output tensor shape must match input tensor shape")
    
    # Get total number of elements
    n = input.numel()
    
    # Launch kernel
    block = 256
    grid = (triton.cdiv(n, block),)
    
    _add_gelu_kernel[grid](input, other, out, n, alpha, BLOCK=block)
    
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
