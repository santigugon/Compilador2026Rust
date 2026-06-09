import torch
import triton
import triton.language as tl

def _clamp(x, min_val, max_val):
    return tl.minimum(tl.maximum(x, min_val), max_val)

@triton.jit
def _logit_kernel(x_ptr, out_ptr, n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Clamp input to [eps, 1 - eps] if eps is not None
    if eps is not None:
        x = _clamp(x, eps, 1.0 - eps)
    
    # Compute logit: log(x / (1 - x))
    # Avoid division by zero by ensuring x is not 0 or 1
    x = tl.where(x == 0.0, 1e-8, x)
    x = tl.where(x == 1.0, 1.0 - 1e-8, x)
    
    y = tl.log(x / (1.0 - x))
    tl.store(out_ptr + offsets, y, mask=mask)


def logit(input, eps=None, *, out=None):
    # Handle scalar input
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    
    # Create output tensor
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape, "Output tensor must have the same shape as input"
        assert out.dtype == input.dtype, "Output tensor must have the same dtype as input"
    
    # Handle special case where eps is None and input is outside [0,1]
    if eps is None:
        # For the case where eps is None, we need to check if input is outside [0,1]
        # This is handled in the kernel with NaN propagation
        pass
    
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Convert eps to a proper Triton constant
    eps_val = eps if eps is not None else None
    
    _logit_kernel[grid](input, out, n, eps_val, BLOCK=block)
    return out
##################################################################################################################################################



import torch

def test_logit():
    results = {}

    # Test case 1: Basic test with input tensor in range [0, 1] without eps
    input1 = torch.tensor([0.2, 0.5, 0.8], device='cuda')
    results["test_case_1"] = logit(input1)

    # Test case 2: Test with input tensor in range [0, 1] with eps
    input2 = torch.tensor([0.0, 0.5, 1.0], device='cuda')
    eps = 1e-6
    results["test_case_2"] = logit(input2, eps=eps)

    # Test case 3: Test with input tensor in range [0, 1] with eps and out tensor
    input3 = torch.tensor([0.1, 0.9], device='cuda')
    out = torch.empty_like(input3)
    results["test_case_3"] = logit(input3, eps=eps, out=out)

    # Test case 4: Test with input tensor in range [0, 1] with out tensor
    input4 = torch.tensor([0.3, 0.7], device='cuda')
    out = torch.empty_like(input4)
    results["test_case_4"] = logit(input4, out=out)

    return results

test_results = test_logit()
