import torch
import triton
import triton.language as tl

@triton.jit
def zeta_kernel(x_ptr, q_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    q = tl.load(q_ptr + offsets, mask=mask)
    
    # Initialize sum
    sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Compute the series sum: sum_{n=0}^{\infty} 1/(n+q)^x
    # We'll use a reasonable number of terms for convergence
    max_terms = 1000
    for n in range(max_terms):
        n_val = tl.full([BLOCK_SIZE], n, dtype=tl.float32)
        denominator = tl.pow(n_val + q, x)
        # Avoid division by zero
        denominator = tl.where(denominator == 0.0, 1.0, denominator)
        term = 1.0 / denominator
        sum += term
        # Check for convergence (optional early exit)
        if tl.all(tl.abs(term) < 1e-10):
            break
    
    tl.store(output_ptr + offsets, sum, mask=mask)

def zeta(input, other, *, out=None):
    if input.shape != other.shape:
        raise ValueError("input and other must have the same shape")
    
    if out is not None and out.shape != input.shape:
        raise ValueError("out must have the same shape as input")
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    
    # Launch kernel
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    zeta_kernel[grid](
        input.data_ptr(),
        other.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

##################################################################################################################################################



import torch

def test_zeta():
    results = {}

    # Test case 1: Basic test with simple values
    input1 = torch.tensor([2.0, 3.0], device='cuda')
    other1 = torch.tensor([1.0, 2.0], device='cuda')
    results["test_case_1"] = zeta(input1, other1)

    # Test case 2: Test with larger values
    input2 = torch.tensor([10.0, 20.0], device='cuda')
    other2 = torch.tensor([5.0, 10.0], device='cuda')
    results["test_case_2"] = zeta(input2, other2)

    # Test case 3: Test with fractional values
    input3 = torch.tensor([2.5, 3.5], device='cuda')
    other3 = torch.tensor([1.5, 2.5], device='cuda')
    results["test_case_3"] = zeta(input3, other3)

    # Test case 4: Test with negative values
    input4 = torch.tensor([-2.0, -3.0], device='cuda')
    other4 = torch.tensor([1.0, 2.0], device='cuda')
    results["test_case_4"] = zeta(input4, other4)

    return results

test_results = test_zeta()
