import torch
import triton
import triton.language as tl

@triton.jit
def svd_reconstruct_kernel(
    A_ptr, U_ptr, S_ptr, Vh_ptr,
    m, n, k,
    stride_A0, stride_A1,
    stride_U0, stride_U1,
    stride_S0,
    stride_Vh0, stride_Vh1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Grid setup
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Load U matrix block
    u_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for i in range(0, BLOCK_SIZE_M, BLOCK_SIZE_K):
        for j in range(0, BLOCK_SIZE_K, BLOCK_SIZE_K):
            if (m_start + i < m) and (j < k):
                for l in range(BLOCK_SIZE_K):
                    if (j + l < k):
                        u_block[i:i+BLOCK_SIZE_K, l] = tl.load(
                            U_ptr + (m_start + i) * stride_U0 + (j + l) * stride_U1
                        )
    
    # Load Vh matrix block
    v_block = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, BLOCK_SIZE_K, BLOCK_SIZE_K):
        for j in range(0, BLOCK_SIZE_N, BLOCK_SIZE_K):
            if (j + j < n) and (i < k):
                for l in range(BLOCK_SIZE_K):
                    if (i + l < k):
                        v_block[l, j:j+BLOCK_SIZE_K] = tl.load(
                            Vh_ptr + (i + l) * stride_Vh0 + (j + l) * stride_Vh1
                        )
    
    # Compute reconstruction: U @ S @ Vh
    for i in range(BLOCK_SIZE_M):
        for j in range(BLOCK_SIZE_N):
            if (m_start + i < m) and (n_start + j < n):
                acc = 0.0
                for l in range(k):
                    s_val = tl.load(S_ptr + l * stride_S0)
                    u_val = tl.load(U_ptr + (m_start + i) * stride_U0 + l * stride_U1)
                    v_val = tl.load(Vh_ptr + l * stride_Vh0 + (n_start + j) * stride_Vh1)
                    acc += u_val * s_val * v_val
                tl.store(
                    A_ptr + (m_start + i) * stride_A0 + (n_start + j) * stride_A1,
                    acc
                )

def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
    # Perform SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # Get dimensions
    m, n = A.shape
    k = S.shape[0]
    
    # Create output tensor
    A_reconstructed = torch.empty_like(A)
    
    # Launch kernel
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    grid = (
        triton.cdiv(m, BLOCK_SIZE_M),
        triton.cdiv(n, BLOCK_SIZE_N)
    )
    
    # Prepare pointers
    stride_A0, stride_A1 = A.stride()
    stride_U0, stride_U1 = U.stride()
    stride_S0 = S.stride(0)
    stride_Vh0, stride_Vh1 = Vh.stride()
    
    # Launch kernel
    svd_reconstruct_kernel[grid](
        A_reconstructed,
        U,
        S,
        Vh,
        m, n, k,
        stride_A0, stride_A1,
        stride_U0, stride_U1,
        stride_S0,
        stride_Vh0, stride_Vh1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return A_reconstructed
