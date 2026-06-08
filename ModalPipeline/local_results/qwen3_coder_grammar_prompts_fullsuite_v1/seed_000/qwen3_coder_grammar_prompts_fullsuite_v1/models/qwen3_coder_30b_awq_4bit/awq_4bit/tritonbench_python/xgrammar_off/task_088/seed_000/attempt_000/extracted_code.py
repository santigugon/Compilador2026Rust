import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, out_ptr, 
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  stride_outm, stride_outn,
                  alpha, beta,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr):
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
    
    out = alpha * accumulator + beta * tl.load(C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    
    out_ptrs = out_ptr + offs_am[:, None] * stride_outm + offs_bn[None, :] * stride_outn
    tl.store(out_ptrs, out)

@triton.jit
def symmetric_kernel(C_ptr, out_ptr, M, N,
                     stride_cm, stride_cn,
                     stride_outm, stride_outn,
                     alpha, beta,
                     BLOCK_SIZE_M: tl.constexpr,
                     BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = BLOCK_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * BLOCK_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, BLOCK_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) // num_pid_n
    pid_n = (pid % num_pid_in_group) % num_pid_n
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    C_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    C_T_ptrs = C_ptr + (offs_bn[:, None] * stride_cm + offs_am[None, :] * stride_cn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, N, BLOCK_SIZE_N):
        c = tl.load(C_ptrs)
        c_T = tl.load(C_T_ptrs)
        accumulator += tl.dot(c, c_T)
    
    out = alpha * accumulator + beta * tl.load(C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    
    out_ptrs = out_ptr + offs_am[:, None] * stride_outm + offs_bn[None, :] * stride_outn
    tl.store(out_ptrs, out)

def matrix_multiply_symmetric(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    M, K = A.shape
    K, N = B.shape
    out = torch.empty(M, N, device=C.device, dtype=C.dtype)
    
    # First operation: C = alpha * torch.mm(A, B) + beta * C
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        A, B, C, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        out.stride(0), out.stride(1),
        alpha, beta,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8
    )
    
    # Second operation: C = alpha * torch.mm(C, C.T) + beta * C
    out2 = torch.empty(M, N, device=C.device, dtype=C.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    symmetric_kernel[grid](
        out, out2,
        M, N,
        out.stride(0), out.stride(1),
        out2.stride(0), out2.stride(1),
        alpha, beta,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64
    )
    
    return out2
