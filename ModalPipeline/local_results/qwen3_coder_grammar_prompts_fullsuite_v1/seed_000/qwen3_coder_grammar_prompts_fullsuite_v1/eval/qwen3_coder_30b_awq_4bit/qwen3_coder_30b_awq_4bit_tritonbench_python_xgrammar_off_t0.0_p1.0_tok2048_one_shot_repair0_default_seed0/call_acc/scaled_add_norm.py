import torch
import triton
import triton.language as tl

@triton.jit
def scaled_add_norm_kernel(y_ptr, x_ptr, alpha, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    y = tl.load(y_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)
    y_new = y + alpha * x
    tl.store(y_ptr + offsets, y_new, mask=mask)
    y_new_sq = y_new * y_new
    tl.store(output_ptr + offsets, y_new_sq, mask=mask)

@triton.jit
def reduce_sum_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    sum_val = tl.sum(input_vals, axis=0)
    tl.store(output_ptr + pid, sum_val, mask=pid < tl.cdiv(n, BLOCK_SIZE))

def scaled_add_norm(y: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    assert y.shape == x.shape, "y and x must have the same shape"
    n = y.numel()
    if n == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=y.device)
    
    # Allocate output tensor for squared values
    output_sq = torch.empty(n, dtype=torch.float32, device=y.device)
    
    # Launch kernel to compute y += alpha * x and store squared values
    BLOCK_SIZE = 1024
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    scaled_add_norm_kernel[(num_blocks,)](y, x, alpha, output_sq, n, BLOCK_SIZE)
    
    # Compute sum of squared values
    sum_output = torch.empty(num_blocks, dtype=torch.float32, device=y.device)
    reduce_sum_kernel[(num_blocks,)](output_sq, sum_output, n, BLOCK_SIZE)
    
    # Reduce to single value
    total_sum = sum_output.sum()
    
    # Return 2-norm
    return torch.sqrt(total_sum)

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
