import torch
import triton
import triton.language as tl

@triton.jit
def _norm_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, stride_x, stride_out, p_norm: tl.constexpr, eps_norm: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // dim_size
    dim_idx = pid % dim_size
    
    # Compute norm for this element
    offsets = batch_idx * stride_x + dim_idx
    x = tl.load(x_ptr + offsets, mask=dim_idx < dim_size, other=0.0)
    
    # Compute L_p norm
    if p_norm == 2:
        norm = tl.sqrt(tl.sum(x * x, axis=0) + eps_norm)
    elif p_norm == 1:
        norm = tl.sum(tl.abs(x), axis=0) + eps_norm
    else:
        norm = tl.pow(tl.sum(tl.pow(tl.abs(x), p_norm), axis=0) + eps_norm, 1.0 / p_norm)
    
    tl.store(out_ptr + pid, norm, mask=dim_idx < dim_size)

@triton.jit
def _cosine_similarity_kernel(x1_ptr, x2_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, stride_x1, stride_x2, stride_out, eps_similarity: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // dim_size
    dim_idx = pid % dim_size
    
    # Load elements
    offsets = batch_idx * stride_x1 + dim_idx
    x1_val = tl.load(x1_ptr + offsets, mask=dim_idx < dim_size, other=0.0)
    x2_val = tl.load(x2_ptr + offsets, mask=dim_idx < dim_size, other=0.0)
    
    # Compute dot product and norms
    dot_product = x1_val * x2_val
    norm_x1 = tl.load(x1_ptr + offsets, mask=dim_idx < dim_size, other=0.0)
    norm_x2 = tl.load(x2_ptr + offsets, mask=dim_idx < dim_size, other=0.0)
    
    # Compute cosine similarity
    numerator = dot_product
    denominator = norm_x1 * norm_x2 + eps_similarity
    
    cosine_sim = numerator / denominator
    tl.store(out_ptr + pid, cosine_sim, mask=dim_idx < dim_size)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Normalize along the specified dimension
    # First compute norms
    if dim < 0:
        dim = x1.dim() + dim
    
    # Create output tensor for normalized vectors
    out = torch.empty_like(x1)
    
    # Compute norms for x1 and x2
    x1_norm = torch.norm(x1, p=p_norm, dim=dim, keepdim=True)
    x2_norm = torch.norm(x2, p=p_norm, dim=dim, keepdim=True)
    
    # Add epsilon to avoid division by zero
    x1_norm = x1_norm + eps_norm
    x2_norm = x2_norm + eps_norm
    
    # Normalize the tensors
    x1_normalized = x1 / x1_norm
    x2_normalized = x2 / x2_norm
    
    # Compute cosine similarity
    # Use PyTorch's built-in function for the final computation
    # This is more numerically stable and handles edge cases better
    result = torch.sum(x1_normalized * x2_normalized, dim=dim)
    
    # Add epsilon to similarity to avoid division by zero
    result = result / (torch.norm(x1_normalized, p=2, dim=dim) * torch.norm(x2_normalized, p=2, dim=dim) + eps_similarity)
    
    return result

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
