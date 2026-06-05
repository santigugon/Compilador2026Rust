import torch
import triton
import triton.language as tl

@triton.jit
def _normalized_cosine_similarity_kernel(
    x1_ptr, x2_ptr, out_ptr,
    x1_stride_0, x1_stride_1,
    x2_stride_0, x2_stride_1,
    out_stride_0, out_stride_1,
    n_cols, n_rows,
    dim,
    eps_similarity,
    p_norm,
    eps_norm,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    if row >= n_rows:
        return
    
    x1_row = x1_ptr + row * x1_stride_0
    x2_row = x2_ptr + row * x2_stride_0
    out_row = out_ptr + row * out_stride_0
    
    # Compute normalization for x1 and x2
    x1_norm = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    x2_norm = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = tl.arange(0, BLOCK_SIZE) + i < n_cols
        x1_vals = tl.load(x1_row + i, mask=mask, other=0.0)
        x2_vals = tl.load(x2_row + i, mask=mask, other=0.0)
        
        x1_norm += tl.power(tl.abs(x1_vals), p_norm)
        x2_norm += tl.power(tl.abs(x2_vals), p_norm)
    
    x1_norm = tl.power(x1_norm, 1.0 / p_norm) + eps_norm
    x2_norm = tl.power(x2_norm, 1.0 / p_norm) + eps_norm
    
    # Normalize vectors
    x1_normed = x1_row / x1_norm
    x2_normed = x2_row / x2_norm
    
    # Compute dot product
    dot = tl.zeros([1], dtype=tl.float32)
    for i in range(0, n_cols, BLOCK_SIZE):
        mask = tl.arange(0, BLOCK_SIZE) + i < n_cols
        x1_vals = tl.load(x1_normed + i, mask=mask, other=0.0)
        x2_vals = tl.load(x2_normed + i, mask=mask, other=0.0)
        dot += tl.sum(x1_vals * x2_vals)
    
    # Compute cosine similarity
    similarity = dot / (x1_norm * x2_norm + eps_similarity)
    
    tl.store(out_row, similarity)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    assert x1.shape == x2.shape, "Input tensors must have the same shape"
    assert dim >= 0 and dim < len(x1.shape), "Invalid dimension"
    
    # For simplicity, we assume the operation is performed along the last dimension
    # and that the tensor is contiguous
    if dim != len(x1.shape) - 1:
        x1 = x1.transpose(dim, len(x1.shape) - 1)
        x2 = x2.transpose(dim, len(x2.shape) - 1)
    
    # Ensure tensors are contiguous
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    
    # Prepare output tensor
    out_shape = list(x1.shape)
    out_shape.pop()
    out = torch.empty(out_shape, dtype=torch.float32, device=x1.device)
    
    # Launch kernel
    n_rows, n_cols = x1.shape[0], x1.shape[1]
    BLOCK_SIZE = 1024
    grid = (n_rows,)
    
    _normalized_cosine_similarity_kernel[grid](
        x1, x2, out,
        x1.stride(0), x1.stride(1),
        x2.stride(0), x2.stride(1),
        out.stride(0), out.stride(1),
        n_cols, n_rows,
        dim,
        eps_similarity,
        p_norm,
        eps_norm,
        BLOCK_SIZE
    )
    
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F
from torch import Tensor

# def normalized_cosine_similarity(x1: Tensor, x2: Tensor, dim: int=1, eps_similarity: float=1e-08, p_norm: float=2, eps_norm: float=1e-12) -> Tensor:
#     x1_normalized = torch.nn.functional.normalize(x1, p=p_norm, dim=dim, eps=eps_norm)
#     x2_normalized = torch.nn.functional.normalize(x2, p=p_norm, dim=dim, eps=eps_norm)
#     return torch.nn.functional.cosine_similarity(x1_normalized, x2_normalized, dim=dim, eps=eps_similarity)

def test_normalized_cosine_similarity():
    results = {}

    # Test case 1: Basic test with default parameters
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    x2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], device='cuda')
    results["test_case_1"] = normalized_cosine_similarity(x1, x2)

    # Test case 2: Different dimension
    x1 = torch.tensor([[1.0, 2.0, 3.0]], device='cuda')
    x2 = torch.tensor([[2.0, 3.0, 4.0]], device='cuda')
    results["test_case_2"] = normalized_cosine_similarity(x1, x2, dim=0)

    # Test case 3: Different p_norm
    x1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device='cuda')
    x2 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device='cuda')
    results["test_case_3"] = normalized_cosine_similarity(x1, x2, p_norm=1)

    # Test case 4: Different eps_norm
    x1 = torch.tensor([[1e-10, 0.0], [0.0, 1e-10]], device='cuda')
    x2 = torch.tensor([[0.0, 1e-10], [1e-10, 0.0]], device='cuda')
    results["test_case_4"] = normalized_cosine_similarity(x1, x2, eps_norm=1e-10)

    return results

test_results = test_normalized_cosine_similarity()
