import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def matmul_kernel(
    A, B, C,
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
    # Create block-level indices
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Compute block offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Load A and B
    A_block = tl.load(A + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_block = tl.load(B + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Compute dot product
    C_block = tl.dot(A_block, B_block)
    
    # Apply activation if needed
    if ACTIVATION == "leaky_relu":
        C_block = tl.where(C_block > 0, C_block, 0.01 * C_block)
    
    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_block = C_block.to(tl.float32)
    tl.store(C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, C_block)

def matmul(input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Handle 1D case (dot product)
    if input.dim() == 1 and other.dim() == 1:
        if out is not None:
            raise ValueError("1D dot product does not support out parameter")
        return torch.dot(input, other)
    
    # Handle 2D case (matrix-matrix product)
    if input.dim() == 2 and other.dim() == 2:
        M, K = input.shape
        K, N = other.shape
        if out is None:
            out = torch.empty(M, N, dtype=torch.float32, device=input.device)
        else:
            assert out.shape == (M, N), "Output tensor shape mismatch"
        
        # Configure grid and block sizes
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 8
        
        # Create grid
        grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
        
        # Launch kernel
        matmul_kernel[grid](
            input, other, out,
            M, N, K,
            input.stride(0), input.stride(1),
            other.stride(0), other.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            GROUP_SIZE_M,
            ACTIVATION="none"
        )
        return out
    
    # Handle batched matrix multiplication
    if input.dim() > 2 and other.dim() > 2:
        # Broadcast dimensions
        batch_dims = torch.broadcast_tensors(input, other)[0].shape[:-2]
        M, K = input.shape[-2:]
        K, N = other.shape[-2:]
        
        # Flatten batch dimensions
        input_flat = input.view(-1, M, K)
        other_flat = other.view(-1, K, N)
        out_flat = torch.empty(input_flat.shape[0], M, N, dtype=torch.float32, device=input.device)
        
        # Process each batch
        for i in range(input_flat.shape[0]):
            matmul_kernel[(1,)](
                input_flat[i], other_flat[i], out_flat[i],
                M, N, K,
                input_flat[i].stride(0), input_flat[i].stride(1),
                other_flat[i].stride(0), other_flat[i].stride(1),
                out_flat[i].stride(0), out_flat[i].stride(1),
                64, 64, 32, 8,
                ACTIVATION="none"
            )
        
        # Reshape output
        out_shape = batch_dims + (M, N)
        return out_flat.view(out_shape)
    
    # Handle matrix-vector product
    if input.dim() == 2 and other.dim() == 1:
        M, K = input.shape
        N = other.shape[0]
        if out is None:
            out = torch.empty(M, dtype=torch.float32, device=input.device)
        else:
            assert out.shape == (M,), "Output tensor shape mismatch"
        
        # Use standard torch matmul for matrix-vector product
        return torch.mm(input, other.unsqueeze(1)).squeeze(1)
    
    # Handle vector-matrix product
    if input.dim() == 1 and other.dim() == 2:
        M = input.shape[0]
        K, N = other.shape
        if out is None:
            out = torch.empty(N, dtype=torch.float32, device=other.device)
        else:
            assert out.shape == (N,), "Output tensor shape mismatch"
        
        # Use standard torch matmul for vector-matrix product
        return torch.mm(input.unsqueeze(0), other).squeeze(0)
    
    # Fallback to torch.matmul for unsupported cases
    if out is None:
        return torch.matmul(input, other)
    else:
        torch.matmul(input, other, out=out)
        return out

##################################################################################################################################################



import torch

def test_matmul():
    results = {}

    # Test case 1: Multiplying two 2D tensors
    tensor1 = torch.tensor([[1, 2], [3, 4]], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([[5, 6], [7, 8]], device='cuda', dtype=torch.float)
    results["test_case_1"] = matmul(tensor1, tensor2)

    # Test case 2: Multiplying a 1D tensor with a 2D tensor
    tensor1 = torch.tensor([1, 2], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([[3, 4], [5, 6]], device='cuda', dtype=torch.float)
    results["test_case_2"] = matmul(tensor1, tensor2)

    # Test case 3: Multiplying a 2D tensor with a 1D tensor
    tensor1 = torch.tensor([[1, 2], [3, 4]], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([5, 6], device='cuda', dtype=torch.float)
    results["test_case_3"] = matmul(tensor1, tensor2)

    # Test case 4: Multiplying two 3D tensors
    tensor1 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device='cuda', dtype=torch.float)
    tensor2 = torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], device='cuda', dtype=torch.float)
    results["test_case_4"] = matmul(tensor1, tensor2)

    return results

test_results = test_matmul()
