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
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) // num_pid_n
    pid_n = (pid % num_pid_in_group) % num_pid_n
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(A_ptrs)
        b = tl.load(B_ptrs)
        accumulator += tl.dot(a, b)
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk
    
    C_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c = accumulator * alpha
    
    if beta != 0.0:
        c += tl.load(C_ptrs) * beta
    
    tl.store(C_ptrs, c)

def matrix_multiply_and_row_dot(A: torch.Tensor, B: torch.Tensor, alpha: float, beta: float, C: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are on the same device and have the correct dtype
    device = A.device
    if B.device != device or C.device != device:
        raise ValueError("All tensors must be on the same device")
    if A.dtype != torch.float32 or B.dtype != torch.float32 or C.dtype != torch.float32:
        raise ValueError("All tensors must be of type torch.float32")
    
    # Get dimensions
    m, k = A.shape
    k2, n = B.shape
    m2, n2 = C.shape
    
    if k != k2 or m != m2 or n != n2:
        raise ValueError("Matrix dimensions are incompatible")
    
    # Create output tensor
    output = torch.empty(m, n, device=device, dtype=torch.float32)
    
    # Launch kernel
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 64, 32
    GROUP_SIZE_M = 8
    
    num_warps = 4
    num_stages = 2
    
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']) * triton.cdiv(n, META['BLOCK_SIZE_N']),
    )
    
    matmul_kernel[grid](
        A, B, output,
        m, n, k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        output.stride(0), output.stride(1),
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages
    )
    
    # Compute dot product of first two rows
    result = torch.dot(output[0], output[1])
    return result
