import torch
import triton
import triton.language as tl

def normalized_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps_similarity: float = 1e-8, p_norm: float = 2, eps_norm: float = 1e-12) -> torch.Tensor:
    # Normalize along the specified dimension
    # First compute the Lp norm for each tensor
    # Then normalize the tensors
    # Finally compute cosine similarity
    
    # Handle negative dimension
    if dim < 0:
        dim = x1.dim() + dim
    
    # Compute Lp norms
    x1_norm = torch.norm(x1, p=p_norm, dim=dim, keepdim=True)
    x2_norm = torch.norm(x2, p=p_norm, dim=dim, keepdim=True)
    
    # Add epsilon to avoid division by zero
    x1_norm = x1_norm + eps_norm
    x2_norm = x2_norm + eps_norm
    
    # Normalize the tensors
    x1_normalized = x1 / x1_norm
    x2_normalized = x2 / x2_norm
    
    # Compute cosine similarity
    # This is the dot product of normalized vectors
    # We need to sum along the specified dimension
    dot_product = torch.sum(x1_normalized * x2_normalized, dim=dim, keepdim=True)
    
    # Add epsilon to similarity to avoid division by zero
    dot_product = dot_product + eps_similarity
    
    # Return the cosine similarity
    return dot_product