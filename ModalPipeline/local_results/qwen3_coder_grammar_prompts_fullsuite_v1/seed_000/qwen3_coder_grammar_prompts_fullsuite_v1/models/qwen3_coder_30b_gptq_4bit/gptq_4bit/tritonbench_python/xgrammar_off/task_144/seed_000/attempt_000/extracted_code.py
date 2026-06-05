import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Get the block indices
    block_idx_m = tl.program_id(0)
    block_idx_n = tl.program_id(1)
    block_idx_k = tl.program_id(2)
    
    # Compute the starting positions for this block
    start_m = block_idx_m * BLOCK_SIZE_M
    start_n = block_idx_n * BLOCK_SIZE_N
    start_k = block_idx_k * BLOCK_SIZE_K
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load A and B tiles
        a = tl.load(A_ptr + start_m * stride_am + k * stride_ak)
        b = tl.load(B_ptr + k * stride_bk + start_n * stride_bn)
        
        # Perform matrix multiplication
        acc += tl.dot(a, b)
    
    # Scale and add to C
    c = tl.load(C_ptr + start_m * stride_cm + start_n * stride_cn)
    acc = acc * alpha + c * beta
    
    # Store the result
    tl.store(C_ptr + start_m * stride_cm + start_n * stride_cn, acc)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are on the same device and have correct dtypes
    assert A.device == B.device == C.device
    assert A.dtype == B.dtype == C.dtype
    
    # Get dimensions
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "Incompatible dimensions between A and B"
    
    # Create output tensor
    output = torch.empty_like(C)
    
    # Launch kernel
    grid = (triton.cdiv(n, 128), triton.cdiv(p, 128), 1)
    matmul_kernel[grid](
        A, B, output,
        n, p, m,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        output.stride(0), output.stride(1),
        alpha, beta,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8
    )
    
    # Compute dot product of first two rows
    result = torch.dot(output[0], output[1])
    
    return result
