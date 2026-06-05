import torch
import triton
import triton.language as tl

@triton.jit
def _normalize_and_compute_cosine_similarity_kernel(
    x1_ptr, x2_ptr, out_ptr,
    x1_stride_0, x1_stride_1,
    x2_stride_0, x2_stride_1,
    out_stride_0, out_stride_1,
    n_cols, dim_size,
    eps_similarity: tl.constexpr,
    eps_norm: tl.constexpr,
    p_norm: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= dim_size:
        return
    
    # Load x1 and x2 for this row
    x1_row = tl.load(x1_ptr + pid * x1_stride_0 + tl.arange(0, BLOCK_SIZE) * x1_stride_1, mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    x2_row = tl.load(x2_ptr + pid * x2_stride_0 + tl.arange(0, BLOCK_SIZE) * x2_stride_1, mask=tl.arange(0, BLOCK_SIZE) < n_cols)
    
    # Compute L_p norm for x1
    if p_norm == 2:
        x1_norm = tl.sqrt(tl.sum(x1_row * x1_row) + eps_norm)
    elif p_norm == 1:
        x1_norm = tl.sum(tl.abs(x1_row)) + eps_norm
    else:
        x1_norm = tl.pow(tl.sum(tl.pow(tl.abs(x1_row), p_norm)) + eps_norm, 1.0 / p_norm)
    
    # Compute L_p norm for x2
    if p_norm == 2:
        x2_norm = tl.sqrt(tl.sum(x2_row * x2_row) + eps_norm)
    elif p_norm == 1:
        x2_norm = tl.sum(tl.abs(x2_row)) + eps_norm
    else:
        x2_norm = tl.pow(tl.sum(tl.pow(tl.abs(x2_row), p_norm)) + eps_norm, 1.0 / p_norm)
    
    # Normalize vectors
    x1_normalized = x1_row / x1_norm
    x2_normalized = x2_row / x2_norm
    
    # Compute dot product
    dot_product = tl.sum(x1_normalized * x2_normalized)
    
    # Compute cosine similarity
    cosine_sim = dot_product / (tl.sqrt(x1_norm * x2_norm) + eps_similarity)
    
    # Store result
    tl.store(out_ptr + pid * out_stride_0 + tl.arange(0, 1) * out_stride_1, cosine_sim)

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    assert x1.shape == x2.shape, "Input tensors must have the same shape"
    assert dim < len(x1.shape), "dim must be less than the number of dimensions of the input tensors"
    
    # Ensure tensors are on the same device and are contiguous
    x1 = x1.contiguous()
    x2 = x2.contiguous()
    
    # Determine the size of the dimension we're computing cosine similarity over
    dim_size = x1.shape[dim]
    n_cols = x1.shape[-1]  # Assuming last dimension is the feature dimension
    
    # Create output tensor
    output_shape = list(x1.shape)
    output_shape.pop(dim)
    out = torch.empty(output_shape, dtype=torch.float32, device=x1.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (dim_size,)
    
    _normalize_and_compute_cosine_similarity_kernel[grid](
        x1_ptr=x1,
        x2_ptr=x2,
        out_ptr=out,
        x1_stride_0=x1.stride(0) if len(x1.shape) > 1 else 1,
        x1_stride_1=x1.stride(1) if len(x1.shape) > 1 else 1,
        x2_stride_0=x2.stride(0) if len(x2.shape) > 1 else 1,
        x2_stride_1=x2.stride(1) if len(x2.shape) > 1 else 1,
        out_stride_0=out.stride(0) if len(out.shape) > 1 else 1,
        out_stride_1=out.stride(1) if len(out.shape) > 1 else 1,
        n_cols=n_cols,
        dim_size=dim_size,
        eps_similarity=eps_similarity,
        eps_norm=eps_norm,
        p_norm=p_norm,
        BLOCK_SIZE=BLOCK_SIZE
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
