import torch
import triton
import triton.language as tl

@triton.jit
def _tensordot_rsqrt_kernel(
    a_ptr, b_ptr, out_ptr,
    a_size, b_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    a_offsets = tl.arange(offset, offset + BLOCK_SIZE)
    b_offsets = tl.arange(offset, offset + BLOCK_SIZE)
    
    a_mask = a_offsets < a_size
    b_mask = b_offsets < b_size
    
    a_vals = tl.load(a_ptr + a_offsets, mask=a_mask)
    b_vals = tl.load(b_ptr + b_offsets, mask=b_mask)
    
    # Element-wise reciprocal square root
    out_vals = 1.0 / tl.sqrt(a_vals * b_vals)
    tl.store(out_ptr + a_offsets, out_vals, mask=a_mask)

def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    # Compute tensordot first
    result = torch.tensordot(a, b, dims)
    
    # Apply reciprocal square root
    return torch.rsqrt(result)

##################################################################################################################################################



import torch

def test_tensordot_rsqrt():
    results = {}

    # Test case 1: Simple contraction with scalar result
    a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    dims = 1
    results["test_case_1"] = tensordot_rsqrt(a, b, dims)

    # Test case 2: Contraction with matrices
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device='cuda')
    dims = ([1], [0])
    results["test_case_2"] = tensordot_rsqrt(a, b, dims)

    # Test case 3: Higher-dimensional tensors
    a = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], device='cuda')
    b = torch.tensor([[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]], device='cuda')
    dims = ([2], [1])
    results["test_case_3"] = tensordot_rsqrt(a, b, dims)

    # Test case 4: Different dimensions for contraction
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    b = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device='cuda')
    dims = ([1], [0])
    results["test_case_4"] = tensordot_rsqrt(a, b, dims)

    return results

test_results = test_tensordot_rsqrt()
