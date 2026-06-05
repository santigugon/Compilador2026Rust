import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    out_ptrs = out_ptr + offs_am[:, None] * stride_out_m + offs_bn[None, :] * stride_out_n
    tl.store(out_ptrs, accumulator, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def matmul(input, other, *, out=None):
    # Handle the case where one or both tensors are 1D
    if input.dim() == 1 and other.dim() == 1:
        # Dot product case
        if out is not None:
            raise NotImplementedError("out parameter not supported for 1D dot product")
        return torch.dot(input, other)
    elif input.dim() == 1 and other.dim() == 2:
        # Matrix-vector product
        if out is not None:
            raise NotImplementedError("out parameter not supported for matrix-vector product")
        return torch.mv(input, other)
    elif input.dim() == 2 and other.dim() == 1:
        # Vector-matrix product
        if out is not None:
            raise NotImplementedError("out parameter not supported for vector-matrix product")
        return torch.mv(other, input.t())
    elif input.dim() == 2 and other.dim() == 2:
        # Matrix-matrix product
        if out is not None:
            # Use PyTorch for out parameter
            return torch.matmul(input, other, out=out)
        else:
            # Use Triton for performance
            m, k = input.shape
            k2, n = other.shape
            if k != k2:
                raise ValueError(f"Input shapes {input.shape} and {other.shape} are incompatible for matmul")
            
            out = torch.empty(m, n, dtype=input.dtype, device=input.device)
            
            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 32
            GROUP_M = 8
            
            num_warps = 4
            num_stages = 2
            
            grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
            
            _matmul_kernel[grid](
                input, other, out,
                m, n, k,
                input.stride(0), input.stride(1),
                other.stride(0), other.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M, BLOCK_N, BLOCK_K,
                GROUP_M,
                num_warps=num_warps,
                num_stages=num_stages
            )
            return out
    else:
        # For higher dimensional tensors, use PyTorch's implementation
        if out is not None:
            return torch.matmul(input, other, out=out)
        else:
            return torch.matmul(input, other)
