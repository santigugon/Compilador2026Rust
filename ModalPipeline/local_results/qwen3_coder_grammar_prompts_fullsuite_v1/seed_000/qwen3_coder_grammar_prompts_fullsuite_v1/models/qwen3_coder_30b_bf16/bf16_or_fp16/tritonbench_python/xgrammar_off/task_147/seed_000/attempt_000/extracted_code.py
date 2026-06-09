import torch
import triton
import triton.language as tl

@triton.jit
def _normalize_kernel(
    x_ptr, 
    output_ptr, 
    n_elements, 
    p_norm, 
    eps_norm,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute norm
    if p_norm == 2.0:
        norm = tl.sqrt(tl.sum(x * x) + eps_norm)
    elif p_norm == 1.0:
        norm = tl.sum(tl.abs(x)) + eps_norm
    else:
        norm = tl.pow(tl.sum(tl.pow(tl.abs(x), p_norm)) + eps_norm, 1.0 / p_norm)
    # Normalize
    normalized = x / norm
    tl.store(output_ptr + offsets, normalized, mask=mask)

@triton.jit
def _pairwise_distance_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    n1,
    n2,
    n_features,
    p_norm,
    eps_distance,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_id = pid // n2
    sample_id = pid % n2
    
    if batch_id >= n1 or sample_id >= n2:
        return
    
    # Load x1 and x2 for this pair
    x1_start = batch_id * n_features
    x2_start = sample_id * n_features
    
    # Compute distance
    if p_norm == 2.0:
        diff = 0.0
        for i in range(n_features):
            diff += (tl.load(x1_ptr + x1_start + i) - tl.load(x2_ptr + x2_start + i)) ** 2
        distance = tl.sqrt(diff + eps_distance)
    elif p_norm == 1.0:
        diff = 0.0
        for i in range(n_features):
            diff += tl.abs(tl.load(x1_ptr + x1_start + i) - tl.load(x2_ptr + x2_start + i))
        distance = diff + eps_distance
    else:
        diff = 0.0
        for i in range(n_features):
            diff += tl.pow(tl.abs(tl.load(x1_ptr + x1_start + i) - tl.load(x2_ptr + x2_start + i)), p_norm)
        distance = tl.pow(diff + eps_distance, 1.0 / p_norm)
    
    output_idx = batch_id * n2 + sample_id
    tl.store(output_ptr + output_idx, distance)

def fused_pairwise_distance_normalize(
    x1: torch.Tensor, 
    x2: torch.Tensor, 
    p_norm: float = 2.0, 
    eps_norm: float = 1e-12, 
    eps_distance: float = 1e-6, 
    keepdim: bool = False
) -> torch.Tensor:
    # Normalize x1 and x2
    x1_normalized = x1.clone()
    x2_normalized = x2.clone()
    
    # Normalize along the last dimension
    n_features = x1.shape[-1]
    n1 = x1.numel() // n_features
    n2 = x2.numel() // n_features
    
    # Create output tensors for normalized data
    x1_normalized = x1_normalized.view(-1, n_features)
    x2_normalized = x2_normalized.view(-1, n_features)
    
    # Normalize x1
    if x1_normalized.numel() > 0:
        x1_flat = x1_normalized.view(-1)
        output_x1 = torch.empty_like(x1_flat)
        grid = (triton.cdiv(x1_flat.numel(), 1024),)
        _normalize_kernel[grid](
            x1_flat,
            output_x1,
            x1_flat.numel(),
            p_norm,
            eps_norm,
            BLOCK_SIZE=1024
        )
        x1_normalized = output_x1.view(x1.shape)
    
    # Normalize x2
    if x2_normalized.numel() > 0:
        x2_flat = x2_normalized.view(-1)
        output_x2 = torch.empty_like(x2_flat)
        grid = (triton.cdiv(x2_flat.numel(), 1024),)
        _normalize_kernel[grid](
            x2_flat,
            output_x2,
            x2_flat.numel(),
            p_norm,
            eps_norm,
            BLOCK_SIZE=1024
        )
        x2_normalized = output_x2.view(x2.shape)
    
    # Compute pairwise distances
    if x1_normalized.numel() == 0 or x2_normalized.numel() == 0:
        return torch.empty(x1.shape[:-1] + x2.shape[:-1], dtype=torch.float, device=x1.device)
    
    # Reshape for pairwise computation
    x1_flat = x1_normalized.view(n1, n_features)
    x2_flat = x2_normalized.view(n2, n_features)
    
    # Allocate output
    output = torch.empty(n1, n2, dtype=torch.float, device=x1.device)
    
    # Compute pairwise distances
    grid = (n1 * n2,)
    _pairwise_distance_kernel[grid](
        x1_flat,
        x2_flat,
        output,
        n1,
        n2,
        n_features,
        p_norm,
        eps_distance,
        BLOCK_SIZE=1024
    )
    
    if keepdim:
        return output.unsqueeze(-1)
    else:
        return output
