import torch
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_kernel(x1_ptr, x2_ptr, out_ptr, n_features: tl.constexpr, n1: tl.constexpr, n2: tl.constexpr, p: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Each block handles one row of the output matrix
    row = pid * BLOCK
    col = pid_j * BLOCK
    
    # Load x1 and x2 for this block
    x1_offsets = row + tl.arange(0, BLOCK)
    x2_offsets = col + tl.arange(0, BLOCK)
    
    # Compute pairwise distances
    for i in range(BLOCK):
        for j in range(BLOCK):
            if i < n_features and j < n_features:
                x1_val = tl.load(x1_ptr + row + i, mask=(row + i) < n1, other=0.0)
                x2_val = tl.load(x2_ptr + col + j, mask=(col + j) < n2, other=0.0)
                diff = x1_val - x2_val
                diff = diff * diff
                tl.atomic_add(out_ptr + (row + i) * n2 + col + j, diff)

@triton.jit
def _normalize_kernel(x_ptr, out_ptr, norm_ptr, n1: tl.constexpr, n2: tl.constexpr, p_norm: tl.constexpr, eps_norm: tl.constexpr, dim_norm: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each block handles one row of the output matrix
    row = pid * BLOCK
    
    # Load the norm values for this row
    norm_offsets = row + tl.arange(0, BLOCK)
    norm_vals = tl.load(norm_ptr + norm_offsets, mask=norm_offsets < n1, other=0.0)
    
    # Normalize the distances
    for i in range(BLOCK):
        if i < n1:
            norm_val = norm_vals[i]
            # Apply normalization
            norm_val = tl.maximum(norm_val, eps_norm)
            # Normalize along the specified dimension
            if dim_norm == 1:
                # Normalize along columns (dim=1)
                for j in range(n2):
                    x_val = tl.load(x_ptr + i * n2 + j, mask=(i * n2 + j) < n1 * n2, other=0.0)
                    x_val = x_val / norm_val
                    tl.store(x_ptr + i * n2 + j, x_val, mask=(i * n2 + j) < n1 * n2)

def normalize_pairwise_distance(x1, x2, p_distance=2.0, eps_distance=1e-6, keepdim=False, p_norm=2, dim_norm=1, eps_norm=1e-12):
    # Ensure inputs are tensors
    if not torch.is_tensor(x1):
        x1 = torch.tensor(x1)
    if not torch.is_tensor(x2):
        x2 = torch.tensor(x2)
    
    # Check shapes
    if x1.shape != x2.shape:
        raise ValueError("x1 and x2 must have the same shape")
    
    # Get dimensions
    n1, n2 = x1.shape[0], x2.shape[0]
    n_features = x1.shape[1]
    
    # Compute pairwise distances
    out = torch.empty(n1, n2, dtype=torch.float32, device=x1.device)
    
    # Compute squared differences
    for i in range(n1):
        for j in range(n2):
            diff = x1[i] - x2[j]
            out[i, j] = torch.sum(diff * diff)
    
    # Apply p_distance norm
    out = torch.pow(out + eps_distance, 1.0 / p_distance)
    
    # Compute normalization values
    norm = torch.norm(x1, p=p_norm, dim=dim_norm, keepdim=True)
    if keepdim:
        norm = norm.expand(n1, n2)
    else:
        # Expand norm to match output shape
        norm = norm.expand(n1, n2)
    
    # Normalize the distances
    out = out / (norm + eps_norm)
    
    return out
