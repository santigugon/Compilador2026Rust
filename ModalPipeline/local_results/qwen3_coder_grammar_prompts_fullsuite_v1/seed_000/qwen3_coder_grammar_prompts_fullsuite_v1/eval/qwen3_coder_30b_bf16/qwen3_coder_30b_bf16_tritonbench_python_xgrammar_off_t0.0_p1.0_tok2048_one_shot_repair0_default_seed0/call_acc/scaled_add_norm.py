import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_norm_kernel(y_ptr, x_ptr, alpha, n, y_out_ptr):
    pid = tl.program_id(0)
    block_size = tl.cdiv(n, tl.num_programs(0))
    offset = pid * block_size
    y_block = tl.load(y_ptr + offset, mask=offset + tl.arange(0, block_size) < n)
    x_block = tl.load(x_ptr + offset, mask=offset + tl.arange(0, block_size) < n)
    y_block = y_block + alpha * x_block
    tl.store(y_ptr + offset, y_block, mask=offset + tl.arange(0, block_size) < n)
    tl.store(y_out_ptr + pid, tl.sum(y_block * y_block))

def scaled_add_norm(y, x, alpha):
    assert y.shape == x.shape, "y and x must have the same shape"
    assert y.dtype == x.dtype, "y and x must have the same dtype"
    n = y.numel()
    y = y.clone()
    y_out = torch.zeros(tl.cdiv(n, 1024), dtype=torch.float32, device=y.device)
    grid = (tl.cdiv(n, 1024),)
    scaled_add_norm_kernel[grid](y, x, alpha, n, y_out)
    return torch.sqrt(y_out.sum())

##################################################################################################################################################



import torch

def test_scaled_add_norm():
    results = {}

    # Test case 1: Basic test with small tensors
    y1 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x1 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha1 = 2.0
    results["test_case_1"] = scaled_add_norm(y1, x1, alpha1).item()

    # Test case 2: Test with negative alpha
    y2 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x2 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha2 = -1.0
    results["test_case_2"] = scaled_add_norm(y2, x2, alpha2).item()

    # Test case 3: Test with zero alpha
    y3 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x3 = torch.tensor([0.5, 0.5, 0.5], device='cuda')
    alpha3 = 0.0
    results["test_case_3"] = scaled_add_norm(y3, x3, alpha3).item()

    # Test case 4: Test with zero vector x
    y4 = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    x4 = torch.tensor([0.0, 0.0, 0.0], device='cuda')
    alpha4 = 2.0
    results["test_case_4"] = scaled_add_norm(y4, x4, alpha4).item()

    return results

test_results = test_scaled_add_norm()
