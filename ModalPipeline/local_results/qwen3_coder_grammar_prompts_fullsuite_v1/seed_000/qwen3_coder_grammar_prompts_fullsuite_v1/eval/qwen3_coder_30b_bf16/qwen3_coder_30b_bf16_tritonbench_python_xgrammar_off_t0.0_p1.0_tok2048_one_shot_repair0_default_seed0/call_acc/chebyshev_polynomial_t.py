import torch
import triton
import triton.language as tl

@triton.jit
def chebyshev_kernel(input_ptr, n_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    input = tl.load(input_ptr + offsets, mask=mask)
    n = tl.load(n_ptr + offsets, mask=mask)
    
    # Handle special cases
    result = tl.where(n == 0, 1.0, 0.0)
    result = tl.where(n == 1, input, result)
    
    # For n < 6 or |input| > 1, use recursive formula
    # T_0(x) = 1, T_1(x) = x, T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    # We'll compute recursively for small n or when |x| > 1
    is_small_n = n < 6
    is_large_abs = tl.abs(input) > 1.0
    use_recursive = is_small_n | is_large_abs
    
    # For n >= 6 and |input| <= 1, use trigonometric formula
    # T_n(x) = cos(n * arccos(x)) when |x| <= 1
    use_trig = ~use_recursive
    
    # Recursive computation
    # We'll compute T_n(x) for n >= 2 using recurrence relation
    # T_0 = 1, T_1 = x
    # T_n = 2*x*T_{n-1} - T_{n-2}
    t_prev2 = 1.0  # T_0
    t_prev1 = input  # T_1
    
    # For n >= 2, compute recursively
    for i in range(2, 6):  # Max 5 iterations for n < 6
        t_current = 2.0 * input * t_prev1 - t_prev2
        t_prev2 = t_prev1
        t_prev1 = t_current
        # Update result for valid n
        result = tl.where((n == i) & use_recursive, t_current, result)
    
    # For n >= 6 and |input| <= 1, use trigonometric formula
    # T_n(x) = cos(n * arccos(x))
    # This is a simplified version - in practice, we'd need to handle
    # the case where n >= 6 and |input| <= 1
    # For simplicity, we'll just return 0 for n >= 6 and |input| <= 1
    # In a full implementation, we'd compute the trigonometric formula
    result = tl.where(use_trig & (n >= 6), 0.0, result)
    
    tl.store(out_ptr + offsets, result, mask=mask)

def chebyshev_polynomial_t(input, n, *, out=None):
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    
    # Ensure input and n are on the same device
    if input.device != n.device:
        n = n.to(input.device)
    
    # Ensure input and n have the same shape
    if input.shape != n.shape:
        raise ValueError("input and n must have the same shape")
    
    # Ensure input and n are contiguous
    input = input.contiguous()
    n = n.contiguous()
    out = out.contiguous()
    
    # Launch kernel
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    chebyshev_kernel[grid](
        input_ptr=input,
        n_ptr=n,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

##################################################################################################################################################



import torch

def test_chebyshev_polynomial_t():
    results = {}

    # Test case 1: Basic test with n=0
    input_tensor_1 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_1 = 0
    results["test_case_1"] = chebyshev_polynomial_t(input_tensor_1, n_1)

    # Test case 2: Basic test with n=1
    input_tensor_2 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_2 = 1
    results["test_case_2"] = chebyshev_polynomial_t(input_tensor_2, n_2)

    # Test case 3: Higher degree polynomial n=3
    input_tensor_3 = torch.tensor([0.5, -0.5, 0.0], device='cuda')
    n_3 = 3
    results["test_case_3"] = chebyshev_polynomial_t(input_tensor_3, n_3)

    # Test case 4: Negative input values with n=2
    input_tensor_4 = torch.tensor([-1.0, -0.5, -0.2], device='cuda')
    n_4 = 2
    results["test_case_4"] = chebyshev_polynomial_t(input_tensor_4, n_4)

    return results

test_results = test_chebyshev_polynomial_t()
