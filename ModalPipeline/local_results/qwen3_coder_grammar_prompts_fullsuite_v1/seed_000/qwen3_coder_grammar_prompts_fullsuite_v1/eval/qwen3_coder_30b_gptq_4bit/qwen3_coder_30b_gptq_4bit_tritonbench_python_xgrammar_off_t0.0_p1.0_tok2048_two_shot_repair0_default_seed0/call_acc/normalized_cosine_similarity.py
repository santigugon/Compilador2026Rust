import torch
import triton
import triton.language as tl

@triton.jit
def _norm_kernel(x_ptr, out_ptr, n: tl.constexpr, dim: tl.constexpr, p: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute L_p norm
    if p == 2:
        x = x * x
        x = tl.sum(x, axis=0)
        x = tl.sqrt(x + eps)
    else:
        x = tl.abs(x) ** p
        x = tl.sum(x, axis=0)
        x = x ** (1.0 / p)
    tl.store(out_ptr + offsets, x, mask=mask)

@triton.jit
def _cosine_similarity_kernel(x1_ptr, x2_ptr, out_ptr, n: tl.constexpr, dim: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    # Compute dot product
    dot = x1 * x2
    dot = tl.sum(dot, axis=0)
    # Compute norms
    x1_norm = tl.sum(x1 * x1, axis=0)
    x2_norm = tl.sum(x2 * x2, axis=0)
    # Normalize
    x1_norm = tl.sqrt(x1_norm + eps)
    x2_norm = tl.sqrt(x2_norm + eps)
    # Compute cosine similarity
    similarity = dot / (x1_norm * x2_norm + eps)
    tl.store(out_ptr + offsets, similarity, mask=mask)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Normalize the tensors along the specified dimension
    # First, we need to compute the L_p norms
    # For simplicity, we'll compute the norms along the specified dimension
    # and then normalize the tensors
    
    # Compute norms
    x1_norm = torch.norm(x1, p=p_norm, dim=dim, keepdim=True)
    x2_norm = torch.norm(x2, p=p_norm, dim=dim, keepdim=True)
    
    # Add epsilon to avoid division by zero
    x1_norm = x1_norm + eps_norm
    x2_norm = x2_norm + eps_norm
    
    # Normalize the tensors
    x1_normalized = x1 / x1_norm
    x2_normalized = x2 / x2_norm
    
    # Compute cosine similarity
    # We'll use a simple approach for now
    # Compute dot product
    dot_product = torch.sum(x1_normalized * x2_normalized, dim=dim, keepdim=True)
    
    # Compute cosine similarity
    cosine_sim = dot_product / (torch.norm(x1_normalized, dim=dim, keepdim=True) * 
                               torch.norm(x2_normalized, dim=dim, keepdim=True) + eps_similarity)
    
    # Return the result
    return cosine_sim.squeeze(dim)

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
