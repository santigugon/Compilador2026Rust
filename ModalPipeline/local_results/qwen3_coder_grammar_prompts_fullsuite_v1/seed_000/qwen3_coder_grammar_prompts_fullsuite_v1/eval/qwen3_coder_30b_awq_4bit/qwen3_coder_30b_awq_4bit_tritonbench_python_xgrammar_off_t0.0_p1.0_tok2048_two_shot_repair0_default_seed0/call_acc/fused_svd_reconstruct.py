import torch
import triton
import triton.language as tl

@triton.jit
def _svd_reconstruct_kernel(u_ptr, s_ptr, vh_ptr, out_ptr, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr, BLOCK: tl.constexpr):
    # Compute reconstruction: U @ diag(s) @ VH
    pid = tl.program_id(0)
    if pid == 0:
        # First kernel: compute U @ diag(s)
        for i in range(m):
            for j in range(k):
                acc = 0.0
                for l in range(k):
                    u_val = tl.load(u_ptr + i * k + l)
                    s_val = tl.load(s_ptr + l)
                    acc += u_val * s_val
                tl.store(out_ptr + i * k + j, acc)
    elif pid == 1:
        # Second kernel: compute (U @ diag(s)) @ VH
        for i in range(m):
            for j in range(n):
                acc = 0.0
                for l in range(k):
                    u_s_val = tl.load(out_ptr + i * k + l)
                    vh_val = tl.load(vh_ptr + l * n + j)
                    acc += u_s_val * vh_val
                tl.store(out_ptr + i * n + j, acc)

def fused_svd_reconstruct(A):
    # Perform SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Reconstruct A using U, S, Vh
    # A_reconstructed = U @ diag(S) @ Vh
    out = torch.empty_like(A)
    
    # For small matrices, use direct computation
    if A.shape[0] * A.shape[1] < 1024:
        # Direct reconstruction
        out = U @ torch.diag(S) @ Vh
        return out
    
    # For larger matrices, use Triton kernel
    m, n = A.shape
    k = S.shape[0]
    
    # Create intermediate tensor for U @ diag(S)
    intermediate = torch.empty(m, k, device=A.device, dtype=A.dtype)
    
    # Launch kernels
    block = 256
    grid1 = (triton.cdiv(m * k, block),)
    grid2 = (triton.cdiv(m * n, block),)
    
    # First kernel: U @ diag(S)
    _svd_reconstruct_kernel[grid1](
        U, S, Vh, intermediate, m, n, k, BLOCK=block
    )
    
    # Second kernel: (U @ diag(S)) @ Vh
    _svd_reconstruct_kernel[grid2](
        intermediate, S, Vh, out, m, n, k, BLOCK=block
    )
    
    return out

##################################################################################################################################################



import torch

def test_fused_svd_reconstruct():
    results = {}
    
    # Test case 1: Square matrix
    A1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
    results["test_case_1"] = fused_svd_reconstruct(A1)
    
    # Test case 2: Rectangular matrix (more rows than columns)
    A2 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')
    results["test_case_2"] = fused_svd_reconstruct(A2)
    
    # Test case 3: Rectangular matrix (more columns than rows)
    A3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
    results["test_case_3"] = fused_svd_reconstruct(A3)
    
    # Test case 4: Single value matrix
    A4 = torch.tensor([[42.0]], device='cuda')
    results["test_case_4"] = fused_svd_reconstruct(A4)
    
    return results

test_results = test_fused_svd_reconstruct()
