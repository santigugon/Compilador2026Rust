import torch
import triton
import triton.language as tl
import math

@triton.jit
def _svd_squeeze_kernel(
    s_ptr, 
    s_out_ptr, 
    n: tl.constexpr, 
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    s = tl.load(s_ptr + offsets, mask=mask, other=0.0)
    # For pseudoinverse, we need to invert non-zero singular values
    # This kernel just copies the singular values for now
    tl.store(s_out_ptr + offsets, s, mask=mask)

@triton.jit
def _svd_invert_kernel(
    s_ptr,
    s_inv_ptr,
    n: tl.constexpr,
    rcond: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    s = tl.load(s_ptr + offsets, mask=mask, other=0.0)
    
    # Compute the threshold
    max_s = tl.max(s)
    threshold = rcond * max_s
    
    # Invert singular values that are above threshold
    s_inv = tl.where(s > threshold, 1.0 / s, 0.0)
    tl.store(s_inv_ptr + offsets, s_inv, mask=mask)

def pseudoinverse_svd(A, *, full_matrices=True, rcond=1e-15, out=None):
    # For simplicity, we'll use PyTorch's SVD implementation and
    # implement the pseudoinverse computation in Triton for the core operations
    
    # Handle scalar input case
    if A.dim() < 2:
        raise ValueError("Input must have at least 2 dimensions")
    
    # Get the last two dimensions
    m, n = A.shape[-2], A.shape[-1]
    
    # For batched operations, we need to handle the batch dimensions
    batch_shape = A.shape[:-2]
    
    # Use PyTorch's SVD for decomposition
    if full_matrices:
        U, S, Vh = torch.linalg.svd(A, full_matrices=True)
    else:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Compute the pseudoinverse using Triton for the inversion part
    # Create output tensor
    if out is not None:
        result = out
    else:
        result = torch.empty_like(A)
    
    # For the pseudoinverse computation: A+ = V * S+ * U^T
    # where S+ is the pseudoinverse of S
    
    # Invert singular values using Triton
    s_inv = torch.empty_like(S)
    n = S.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # We need to reshape S to 1D for Triton kernel
    S_flat = S.view(-1)
    s_inv_flat = s_inv.view(-1)
    
    # Apply inversion kernel
    _svd_invert_kernel[grid](S_flat, s_inv_flat, n, rcond, BLOCK=block)
    
    # Reshape back to original shape
    s_inv = s_inv_flat.view(S.shape)
    
    # Compute pseudoinverse: V @ diag(s_inv) @ U.T
    # For batched operations, we need to handle each batch separately
    if len(batch_shape) == 0:
        # Single matrix case
        result = Vh.mH @ torch.diag(s_inv) @ U.mH
    else:
        # Batched case
        # Reshape for batched matrix multiplication
        batch_size = torch.prod(torch.tensor(batch_shape))
        U = U.view(batch_size, m, m)
        Vh = Vh.view(batch_size, n, n)
        s_inv = s_inv.view(batch_size, min(m, n))
        
        # Compute pseudoinverse for each batch
        result_batch = torch.empty(batch_size, n, m, dtype=A.dtype, device=A.device)
        for i in range(batch_size):
            result_batch[i] = Vh[i].mH @ torch.diag(s_inv[i]) @ U[i].mH
            
        # Reshape back to original batch shape
        result = result_batch.view(*batch_shape, n, m)
    
    # If out is provided, copy result to it
    if out is not None:
        out.copy_(result)
        return out
    
    return result

##################################################################################################################################################



import torch

def test_pseudoinverse_svd():
    results = {}

    # Test case 1: Square matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = pseudoinverse_svd(A1)

    # Test case 4: Singular matrix
    A4 = torch.tensor([[1.0, 2.0], [2.0, 4.0]], device='cuda')
    results["test_case_4"] = pseudoinverse_svd(A4)

    return results

test_results = test_pseudoinverse_svd()
