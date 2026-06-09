import torch
import triton
import triton.language as tl

@triton.jit
def _norm_kernel(x_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, stride_x, stride_out, p_norm: tl.constexpr, eps_norm: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    dim_offset = pid * dim_size
    offsets = dim_offset + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if p_norm == 2:
        x = x * x
        x = tl.sum(x, axis=0)
        x = tl.sqrt(x + eps_norm)
    else:
        x = tl.abs(x) ** p_norm
        x = tl.sum(x, axis=0)
        x = x ** (1.0 / p_norm) + eps_norm
    tl.store(out_ptr + pid, x, mask=pid < dim_size)

@triton.jit
def _cosine_similarity_kernel(x1_ptr, x2_ptr, out_ptr, n: tl.constexpr, dim_size: tl.constexpr, stride_x1, stride_x2, stride_out, eps_similarity: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    dim_offset = pid * dim_size
    offsets = dim_offset + tl.arange(0, BLOCK)
    mask = offsets < n
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    dot = x1 * x2
    dot = tl.sum(dot, axis=0)
    x1_norm = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2_norm = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    x1_norm = tl.sum(x1_norm * x1_norm, axis=0)
    x2_norm = tl.sum(x2_norm * x2_norm, axis=0)
    x1_norm = tl.sqrt(x1_norm + eps_similarity)
    x2_norm = tl.sqrt(x2_norm + eps_similarity)
    norm_product = x1_norm * x2_norm
    cosine = dot / norm_product
    tl.store(out_ptr + pid, cosine, mask=pid < dim_size)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Normalize along the specified dimension
    # First, compute the norms
    if dim < 0:
        dim = x1.dim() + dim
    
    # Compute the size of the dimension we're normalizing along
    dim_size = x1.size(dim)
    # Compute the total number of elements
    n = x1.numel()
    
    # Compute the stride for the dimension we're normalizing along
    stride_x1 = x1.stride(dim)
    stride_x2 = x2.stride(dim)
    
    # Compute the output size
    out_size = list(x1.shape)
    out_size.pop(dim)
    out = torch.empty(out_size, dtype=torch.float32, device=x1.device)
    
    # Compute the block size
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Compute norms
    x1_norm = torch.empty_like(x1)
    x2_norm = torch.empty_like(x2)
    
    # Normalize x1 and x2
    x1_norm = x1 / (torch.norm(x1, p=p_norm, dim=dim, keepdim=True) + eps_norm)
    x2_norm = x2 / (torch.norm(x2, p=p_norm, dim=dim, keepdim=True) + eps_norm)
    
    # Compute cosine similarity
    dot_product = torch.sum(x1_norm * x2_norm, dim=dim)
    norm_product = torch.norm(x1_norm, p=2, dim=dim) * torch.norm(x2_norm, p=2, dim=dim)
    cosine_similarity = dot_product / (norm_product + eps_similarity)
    
    return cosine_similarity

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
