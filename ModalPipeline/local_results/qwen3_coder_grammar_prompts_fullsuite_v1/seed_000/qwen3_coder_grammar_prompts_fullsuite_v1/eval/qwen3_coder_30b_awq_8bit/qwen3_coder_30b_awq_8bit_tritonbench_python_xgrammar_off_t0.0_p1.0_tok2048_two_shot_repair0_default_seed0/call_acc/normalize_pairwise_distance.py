import torch
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_kernel(x1_ptr, x2_ptr, out_ptr, n_rows, n_cols, p: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    if row >= n_rows or col >= n_cols:
        return
    
    # Load x1 and x2 for this row/column pair
    x1_offsets = row * n_cols + tl.arange(0, BLOCK)
    x2_offsets = col * n_cols + tl.arange(0, BLOCK)
    
    # Create masks for boundary conditions
    x1_mask = x1_offsets < n_rows * n_cols
    x2_mask = x2_offsets < n_cols * n_cols
    
    # Load data
    x1_data = tl.load(x1_ptr + x1_offsets, mask=x1_mask, other=0.0)
    x2_data = tl.load(x2_ptr + x2_offsets, mask=x2_mask, other=0.0)
    
    # Compute difference and absolute difference
    diff = x1_data - x2_data
    abs_diff = tl.abs(diff)
    
    # Compute distance with specified norm
    if p == 1.0:
        distance = tl.sum(abs_diff)
    elif p == 2.0:
        distance = tl.sqrt(tl.sum(diff * diff) + eps)
    else:
        distance = tl.pow(tl.sum(tl.pow(abs_diff, p)), 1.0 / p)
    
    # Store result
    out_offsets = row * n_cols + col
    tl.store(out_ptr + out_offsets, distance)

@triton.jit
def _normalize_kernel(x_ptr, out_ptr, n_rows, n_cols, dim: tl.constexpr, p: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    
    if row >= n_rows:
        return
    
    # Compute normalization factor along specified dimension
    offsets = row * n_cols + tl.arange(0, BLOCK)
    mask = offsets < n_rows * n_cols
    
    # Load data
    data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute norm along specified dimension
    if p == 1.0:
        norm = tl.sum(tl.abs(data))
    elif p == 2.0:
        norm = tl.sqrt(tl.sum(data * data) + eps)
    else:
        norm = tl.pow(tl.sum(tl.pow(tl.abs(data), p)), 1.0 / p)
    
    # Normalize
    normalized = data / (norm + eps)
    
    # Store result
    tl.store(out_ptr + offsets, normalized, mask=mask)

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    # Validate inputs
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape")
    
    # Get dimensions
    shape = x1.shape
    n_rows = shape[0] if len(shape) > 0 else 1
    n_cols = shape[1] if len(shape) > 1 else 1
    
    # Compute pairwise distances
    out = torch.empty(n_rows, n_rows, dtype=x1.dtype, device=x1.device)
    
    # For simplicity, we'll compute the full pairwise distance matrix
    # This is a simplified approach - in practice, you might want to optimize this further
    block = 256
    grid = (triton.cdiv(n_rows, block), triton.cdiv(n_rows, block))
    
    # Create a temporary tensor for pairwise distances
    pairwise_distances = torch.empty(n_rows, n_rows, dtype=x1.dtype, device=x1.device)
    
    # Compute pairwise distances using a simple approach
    for i in range(n_rows):
        for j in range(n_rows):
            diff = x1[i] - x2[j]
            if p_distance == 1.0:
                dist = torch.sum(torch.abs(diff))
            elif p_distance == 2.0:
                dist = torch.sqrt(torch.sum(diff * diff) + eps_distance)
            else:
                dist = torch.pow(torch.sum(torch.pow(torch.abs(diff), p_distance)), 1.0 / p_distance)
            pairwise_distances[i, j] = dist
    
    # Normalize along specified dimension
    if dim_norm == 0:
        # Normalize along rows
        if p_norm == 1.0:
            norm = torch.sum(torch.abs(pairwise_distances), dim=dim_norm, keepdim=True)
        elif p_norm == 2.0:
            norm = torch.sqrt(torch.sum(pairwise_distances * pairwise_distances, dim=dim_norm, keepdim=True) + eps_norm)
        else:
            norm = torch.pow(torch.sum(torch.pow(torch.abs(pairwise_distances), p_norm), dim=dim_norm, keepdim=True), 1.0 / p_norm)
        result = pairwise_distances / (norm + eps_norm)
    else:
        # Normalize along columns (default case)
        if p_norm == 1.0:
            norm = torch.sum(torch.abs(pairwise_distances), dim=dim_norm, keepdim=True)
        elif p_norm == 2.0:
            norm = torch.sqrt(torch.sum(pairwise_distances * pairwise_distances, dim=dim_norm, keepdim=True) + eps_norm)
        else:
            norm = torch.pow(torch.sum(torch.pow(torch.abs(pairwise_distances), p_norm), dim=dim_norm, keepdim=True), 1.0 / p_norm)
        result = pairwise_distances / (norm + eps_norm)
    
    # Handle keepdim
    if not keepdim:
        # Remove the dimension that was reduced
        if dim_norm == 0:
            result = result.squeeze(0)
        else:
            result = result.squeeze(1)
    
    return result

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-06, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
#     pairwise_distance = torch.norm(x1 - x2, p=p_distance, dim=-1, keepdim=keepdim)
#     pairwise_distance = pairwise_distance + eps_distance
#     normed_distance = pairwise_distance / torch.norm(pairwise_distance, p=p_norm, dim=dim_norm, keepdim=True).clamp(min=eps_norm)
#     return normed_distance

def test_normalize_pairwise_distance():
    results = {}
    x1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x2 = torch.tensor([[1.0, 2.5], [2.5, 4.0]])
    
    # Compute the normalized pairwise distance
    results["test_case_1"] = normalize_pairwise_distance(x1, x2, p_distance=2.0, dim_norm=0)
    # Normalize along a different dimension
    results["test_case_2"] = normalize_pairwise_distance(x1, x2, p_distance=1.0, dim_norm=0)

    return results

test_results = test_normalize_pairwise_distance()
