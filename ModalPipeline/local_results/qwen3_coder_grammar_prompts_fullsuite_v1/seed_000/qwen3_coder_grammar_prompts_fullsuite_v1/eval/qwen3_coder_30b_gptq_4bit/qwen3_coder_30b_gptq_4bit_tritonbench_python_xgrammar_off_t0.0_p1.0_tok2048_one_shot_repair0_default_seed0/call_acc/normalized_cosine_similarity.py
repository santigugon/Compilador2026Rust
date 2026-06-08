import torch
import triton
import triton.language as tl

@triton.jit
def _normalized_cosine_similarity_kernel(
    x1_ptr, x2_ptr, out_ptr,
    x1_stride, x2_stride, out_stride,
    dim_size, other_size,
    eps_similarity: tl.constexpr,
    eps_norm: tl.constexpr,
    p_norm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_pos = block_idx * BLOCK_SIZE
    
    # Load x1 and x2 data
    x1 = tl.load(x1_ptr + start_pos, mask=start_pos + tl.arange(0, BLOCK_SIZE) < dim_size)
    x2 = tl.load(x2_ptr + start_pos, mask=start_pos + tl.arange(0, BLOCK_SIZE) < dim_size)
    
    # Compute L_p norm
    norm_x1 = tl.sum(x1 * x1, axis=0) ** (1.0 / p_norm) + eps_norm
    norm_x2 = tl.sum(x2 * x2, axis=0) ** (1.0 / p_norm) + eps_norm
    
    # Compute dot product
    dot_product = tl.sum(x1 * x2, axis=0)
    
    # Compute cosine similarity
    similarity = dot_product / (norm_x1 * norm_x2 + eps_similarity)
    
    # Store result
    tl.store(out_ptr + block_idx, similarity)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Ensure inputs are on the same device and have the same shape
    assert x1.device == x2.device, "x1 and x2 must be on the same device"
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    
    # Normalize along the specified dimension
    # Compute L_p norms
    norm_x1 = torch.norm(x1, p=p_norm, dim=dim, keepdim=True) + eps_norm
    norm_x2 = torch.norm(x2, p=p_norm, dim=dim, keepdim=True) + eps_norm
    
    # Normalize the tensors
    x1_normalized = x1 / norm_x1
    x2_normalized = x2 / norm_x2
    
    # Compute cosine similarity
    dot_product = torch.sum(x1_normalized * x2_normalized, dim=dim)
    
    # Apply epsilon to avoid division by zero
    similarity = dot_product / (norm_x1.squeeze(dim) * norm_x2.squeeze(dim) + eps_similarity)
    
    return similarity

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
