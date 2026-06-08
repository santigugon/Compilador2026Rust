import torch
import triton
import triton.language as tl

@triton.jit
def _svd_reconstruct_kernel(
    u_ptr, s_ptr, vh_ptr,
    out_ptr,
    m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    u_stride_0: tl.constexpr, u_stride_1: tl.constexpr,
    s_stride_0: tl.constexpr,
    vh_stride_0: tl.constexpr, vh_stride_1: tl.constexpr,
    out_stride_0: tl.constexpr, out_stride_1: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    block_size = BLOCK
    start = pid * block_size
    end = min(start + block_size, m)
    
    # Process each row
    for i in range(start, end):
        # For each row, compute the reconstruction
        for j in range(n):
            acc = 0.0
            for l in range(k):
                u_val = tl.load(u_ptr + i * u_stride_0 + l * u_stride_1)
                s_val = tl.load(s_ptr + l * s_stride_0)
                vh_val = tl.load(vh_ptr + l * vh_stride_0 + j * vh_stride_1)
                acc += u_val * s_val * vh_val
            tl.store(out_ptr + i * out_stride_0 + j * out_stride_1, acc)

def fused_svd_reconstruct(A):
    # Compute SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Get dimensions
    m, n = A.shape
    k = S.shape[0]
    
    # Create output tensor
    out = torch.empty_like(A)
    
    # Set up kernel launch parameters
    BLOCK = 32
    grid = (triton.cdiv(m, BLOCK),)
    
    # Launch kernel
    _svd_reconstruct_kernel[grid](
        U, S, Vh,
        out,
        m, n, k,
        U.stride(0), U.stride(1),
        S.stride(0),
        Vh.stride(0), Vh.stride(1),
        out.stride(0), out.stride(1),
        BLOCK=BLOCK
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
