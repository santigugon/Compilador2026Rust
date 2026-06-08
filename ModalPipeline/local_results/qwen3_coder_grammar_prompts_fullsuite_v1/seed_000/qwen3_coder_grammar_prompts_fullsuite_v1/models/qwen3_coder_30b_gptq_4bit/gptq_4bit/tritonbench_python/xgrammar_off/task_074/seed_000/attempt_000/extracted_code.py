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
        x = x ** (1.0 / p) + eps
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
    # First, we need to compute the L_p norms of both tensors
    # Then we normalize the tensors
    # Finally, we compute the cosine similarity
    
    # Handle the case where one of the tensors is a scalar
    if x1.dim() == 0:
        x1 = x1.unsqueeze(0)
    if x2.dim() == 0:
        x2 = x2.unsqueeze(0)
    
    # Ensure both tensors have the same shape
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape")
    
    # Compute the L_p norm for both tensors
    # For simplicity, we'll compute the norm along the specified dimension
    # and then normalize the tensors
    
    # Create output tensor
    out = torch.empty_like(x1)
    
    # Get the size of the tensor along the specified dimension
    n = x1.numel()
    
    # Compute the L_p norm along the specified dimension
    # We'll use a simple approach for now
    if dim < 0:
        dim = x1.dim() + dim
    
    # For this implementation, we'll compute the cosine similarity directly
    # by normalizing the tensors and computing the dot product
    
    # Normalize the tensors
    x1_norm = torch.norm(x1, p=p_norm, dim=dim, keepdim=True)
    x2_norm = torch.norm(x2, p=p_norm, dim=dim, keepdim=True)
    
    # Add epsilon to avoid division by zero
    x1_norm = x1_norm + eps_norm
    x2_norm = x2_norm + eps_norm
    
    # Normalize the tensors
    x1_normalized = x1 / x1_norm
    x2_normalized = x2 / x2_norm
    
    # Compute cosine similarity
    # We'll compute the dot product along the specified dimension
    # and divide by the product of the norms
    dot_product = torch.sum(x1_normalized * x2_normalized, dim=dim, keepdim=True)
    norm_product = x1_norm * x2_norm
    
    # Compute cosine similarity
    similarity = dot_product / (norm_product + eps_similarity)
    
    # Return the result
    return similarity.squeeze(dim) if dim < x1.dim() else similarity
