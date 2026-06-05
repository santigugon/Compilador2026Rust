import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    remaining = pid % num_pid_in_group
    pid_m = first_pid_m + (remaining % GROUP_M)
    pid_n = remaining // GROUP_M

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def matmul(input, other, *, out=None):
    if input.dim() == 1 and other.dim() == 1:
        # 1D dot product
        if out is not None:
            raise RuntimeError("1D dot product does not support out parameter")
        return torch.dot(input, other)
    elif input.dim() == 1 and other.dim() == 2:
        # 1D x 2D -> matrix-vector product
        if out is not None:
            raise RuntimeError("1D x 2D matmul does not support out parameter")
        return torch.mv(input, other)
    elif input.dim() == 2 and other.dim() == 1:
        # 2D x 1D -> matrix-vector product
        if out is not None:
            raise RuntimeError("2D x 1D matmul does not support out parameter")
        return torch.mv(input, other)
    elif input.dim() == 2 and other.dim() == 2:
        # 2D x 2D -> matrix-matrix product
        if out is not None:
            out = out.contiguous()
        else:
            out = torch.empty(input.size(0), other.size(1), dtype=input.dtype, device=input.device)
        
        M, K = input.shape
        K, N = other.shape
        
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        GROUP_M = 8
        
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        _matmul_kernel[grid](
            input, other, out,
            M, N, K,
            input.stride(0), input.stride(1),
            other.stride(0), other.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K,
            GROUP_M
        )
        return out
    else:
        # N-dimensional tensors (batched matrix multiply)
        if out is not None:
            out = out.contiguous()
        else:
            out = torch.empty(*input.shape[:-1], other.shape[-1], dtype=input.dtype, device=input.device)
        
        # Handle batch dimensions
        batch_shape = torch.broadcast_shapes(input.shape[:-2], other.shape[:-2])
        batch_size = 1
        for dim in batch_shape:
            batch_size *= dim
            
        # Reshape tensors for batched operation
        input_reshaped = input.view(-1, input.shape[-2], input.shape[-1])
        other_reshaped = other.view(-1, other.shape[-2], other.shape[-1])
        out_reshaped = out.view(-1, out.shape[-2], out.shape[-1])
        
        # Process each batch
        for i in range(batch_size):
            M, K = input_reshaped[i].shape
            K, N = other_reshaped[i].shape
            
            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 32
            GROUP_M = 8
            
            grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
            
            _matmul_kernel[grid](
                input_reshaped[i], other_reshaped[i], out_reshaped[i],
                M, N, K,
                input_reshaped[i].stride(0), input_reshaped[i].stride(1),
                other_reshaped[i].stride(0), other_reshaped[i].stride(1),
                out_reshaped[i].stride(0), out_reshaped[i].stride(1),
                BLOCK_M, BLOCK_N, BLOCK_K,
                GROUP_M
            )
        
        return out
