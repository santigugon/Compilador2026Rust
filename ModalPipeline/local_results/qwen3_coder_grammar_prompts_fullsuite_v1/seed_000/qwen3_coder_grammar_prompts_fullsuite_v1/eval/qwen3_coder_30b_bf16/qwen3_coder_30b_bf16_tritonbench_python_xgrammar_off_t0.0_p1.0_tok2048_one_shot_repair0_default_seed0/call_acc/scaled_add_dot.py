import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_dot_kernel(y_ptr, x_ptr, out_ptr, n, alpha, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < n
    y = tl.load(y_ptr + offset, mask=mask)
    x = tl.load(x_ptr + offset, mask=mask)
    y_new = y + alpha * x
    tl.store(y_ptr + offset, y_new, mask=mask)
    # Compute dot product of modified y with itself
    dot_product = tl.sum(y_new * y_new)
    if pid == 0:
        tl.store(out_ptr, dot_product)

def scaled_add_dot(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.shape == x.shape, "y and x must have the same shape"
    assert y.dtype == x.dtype, "y and x must have the same dtype"
    n = y.numel()
    out = torch.empty(1, dtype=y.dtype, device=y.device)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    scaled_add_dot_kernel[grid](y, x, out, n, alpha, BLOCK_SIZE=BLOCK_SIZE)
    return out

##################################################################################################################################################



import torch
from torch import Tensor

def test_scaled_add_dot():
    results = {}

    # Test case 1: Basic functionality
    y1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha1 = 2.0
    results["test_case_1"] = scaled_add_dot(y1, x1, alpha1).item()

    # Test case 2: Zero tensor x
    y2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x2 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    alpha2 = 2.0
    results["test_case_2"] = scaled_add_dot(y2, x2, alpha2).item()

    # Test case 3: Zero tensor y
    y3 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    x3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    alpha3 = 1.0
    results["test_case_3"] = scaled_add_dot(y3, x3, alpha3).item()

    # Test case 4: Negative alpha
    y4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x4 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha4 = -1.0
    results["test_case_4"] = scaled_add_dot(y4, x4, alpha4).item()

    return results

test_results = test_scaled_add_dot()
