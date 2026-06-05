import torch
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(
    input_ptr, 
    other_ptr, 
    output_ptr,
    alpha,
    approximate,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    
    # Add input and scaled other
    scaled_other = alpha * other
    sum_val = input + scaled_other
    
    # Apply GELU
    if approximate == "none":
        # Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt_2 = 1.4142135623730951
        erf_arg = sum_val / sqrt_2
        erf_val = tl.math.erf(erf_arg)
        gelu_val = sum_val * 0.5 * (1.0 + erf_val)
    else:
        # Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        pi = 3.141592653589793
        sqrt_2_over_pi = 0.7978845608028654
        x_cubed = sum_val * sum_val * sum_val
        tanh_arg = sqrt_2_over_pi * (sum_val + 0.044715 * x_cubed)
        tanh_val = tl.math.tanh(tanh_arg)
        gelu_val = 0.5 * sum_val * (1.0 + tanh_val)
    
    tl.store(output_ptr + offsets, gelu_val, mask=mask)

def add_gelu(input, other, alpha=1, approximate='none', out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    # Ensure input and other have the same dtype
    if isinstance(other, (int, float)):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    else:
        other = other.to(input.dtype).to(input.device)
    
    # Flatten tensors to 1D for kernel processing
    input_flat = input.flatten()
    other_flat = other.flatten()
    out_flat = out.flatten()
    
    # Ensure tensors have the same size
    assert input_flat.numel() == other_flat.numel(), "Input and other must have the same number of elements"
    
    n_elements = input_flat.numel()
    BLOCK_SIZE = 1024
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Determine approximate parameter for kernel
    approx_param = "none" if approximate == "none" else "tanh"
    
    gelu_kernel[grid](
        input_flat,
        other_flat,
        out_flat,
        alpha,
        approx_param,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.reshape(input.shape)

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
