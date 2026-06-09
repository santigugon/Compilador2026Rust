import torch
import triton
import triton.language as tl

@triton.jit
def bessel_j1_kernel(x_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    """
    Compute Bessel function of the first kind of order 1 (J1) for each element
    using series expansion.
    """
    # Compute the global thread index
    pid = tl.program_id(0)
    # Compute the starting index for this block
    block_start = pid * BLOCK_SIZE
    # Create a mask to handle boundary conditions
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Initialize result
    j1 = tl.zeros_like(x)
    
    # Series expansion for J1(x) = (x/2) * \sum_{k=0}^{\infty} \frac{(-1)^k (x/2)^{2k}}{k! (k+1)!}
    # We'll compute a few terms for reasonable accuracy
    # For small x, we can use the series directly
    # For large x, we can use asymptotic expansion, but for simplicity we'll use series
    
    # Precompute x/2
    x_half = x * 0.5
    
    # Initialize terms
    term = x_half
    sign = 1.0
    k = 1
    
    # Compute first few terms of the series
    # We'll use 10 terms for reasonable accuracy
    for _ in range(10):
        j1 += sign * term
        term *= -x_half * x_half / (k * (k + 1))
        sign *= -1.0
        k += 1
    
    # Store result
    tl.store(output_ptr + offsets, j1, mask=mask)


def bessel_j1(input, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    
    # Get the total number of elements
    n = input.numel()
    
    # Launch kernel
    grid = (triton.cdiv(n, 1024),)
    bessel_j1_kernel[grid](input, out, n, BLOCK_SIZE=1024)
    
    return out
##################################################################################################################################################



import torch

def test_bessel_j1():
    results = {}

    # Test case 1: Basic test with a single positive value
    input1 = torch.tensor([1.0], device='cuda')
    results["test_case_1"] = bessel_j1(input1)

    # Test case 2: Test with a tensor of multiple values
    input2 = torch.tensor([0.0, 1.0, 2.0, 3.0], device='cuda')
    results["test_case_2"] = bessel_j1(input2)

    # Test case 3: Test with a tensor of negative values
    input3 = torch.tensor([-1.0, -2.0, -3.0], device='cuda')
    results["test_case_3"] = bessel_j1(input3)

    # Test case 4: Test with a larger tensor
    input4 = torch.linspace(-5.0, 5.0, steps=10, device='cuda')
    results["test_case_4"] = bessel_j1(input4)

    return results

test_results = test_bessel_j1()
