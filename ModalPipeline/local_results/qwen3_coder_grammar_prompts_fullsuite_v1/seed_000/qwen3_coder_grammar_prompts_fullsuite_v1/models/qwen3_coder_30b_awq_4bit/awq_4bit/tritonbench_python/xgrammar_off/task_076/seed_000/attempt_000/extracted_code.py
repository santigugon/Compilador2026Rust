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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
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
        accumulator = tl.dot(a, b, accumulator)
        A_ptrs += BLOCK_SIZE_K * stride_ak
        B_ptrs += BLOCK_SIZE_K * stride_bk
    
    if ACTIVATION == "leaky_relu":
        accumulator = tl.leaky_relu(accumulator)
    
    C_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    tl.store(C_ptrs, accumulator)

def matmul(input, other, *, out=None):
    if input.dim() == 1 and other.dim() == 1:
        if out is not None:
            raise ValueError("1D dot product does not support out parameter")
        return torch.dot(input, other)
    elif input.dim() == 1 and other.dim() == 2:
        if out is not None:
            raise ValueError("Matrix-vector product does not support out parameter")
        return torch.mv(input, other)
    elif input.dim() == 2 and other.dim() == 1:
        if out is not None:
            raise ValueError("Matrix-vector product does not support out parameter")
        return torch.mv(other, input)
    elif input.dim() == 2 and other.dim() == 2:
        if out is not None:
            if out.shape != (input.size(0), other.size(1)):
                raise ValueError("Output tensor shape mismatch")
            out = torch.empty(input.size(0), other.size(1), dtype=input.dtype, device=input.device)
        else:
            out = torch.empty(input.size(0), other.size(1), dtype=input.dtype, device=input.device)
        
        M, K = input.shape
        K, N = other.shape
        
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 8
        
        num_warps = 4
        if K > 1024:
            num_warps = 8
        if K > 2048:
            num_warps = 16
            
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        
        matmul_kernel[grid](
            input, other, out,
            M, N, K,
            input.stride(0), input.stride(1),
            other.stride(0), other.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            ACTIVATION=None
        )
        return out
    else:
        # For higher dimensional tensors, use torch's built-in matmul
        if out is not None:
            out = torch.matmul(input, other, out=out)
        else:
            out = torch.matmul(input, other)
        return out
