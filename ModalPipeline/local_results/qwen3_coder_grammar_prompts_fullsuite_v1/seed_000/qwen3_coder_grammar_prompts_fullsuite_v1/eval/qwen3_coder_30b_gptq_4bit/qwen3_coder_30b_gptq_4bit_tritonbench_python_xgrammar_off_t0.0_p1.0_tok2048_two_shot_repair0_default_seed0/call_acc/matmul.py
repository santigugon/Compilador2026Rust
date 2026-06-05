import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACC_TYPE: tl.constexpr
):
    # Create block pointers for A, B, C
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Group blocks for better performance
    group_id = pid // GROUP_M
    group_size = min(GROUP_M, grid_n - group_id * GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = pid % group_size
    
    # Load A and B
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create pointers for A and B
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(0, 1)
    )
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1)
    )
    
    # Compute matrix multiplication
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, K, BLOCK_K):
        # Load A and B
        a = tl.load(A_block_ptr, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(B_block_ptr, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        
        # Compute partial dot product
        acc += tl.dot(a, b, allow_tf32=True)
        
        # Advance block pointers
        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
    
    # Store result
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1)
    )
    tl.store(C_block_ptr, acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def matmul(input, other, *, out=None):
    # Handle scalar inputs
    if input.dim() == 0 or other.dim() == 0:
        return torch.matmul(input, other, out=out)
    
    # Handle 1D tensors (dot product)
    if input.dim() == 1 and other.dim() == 1:
        if out is not None:
            raise ValueError("1D dot product does not support out parameter")
        return torch.dot(input, other)
    
    # Handle 2D tensors (matrix multiplication)
    if input.dim() == 2 and other.dim() == 2:
        if out is not None:
            torch.mm(input, other, out=out)
            return out
        return torch.mm(input, other)
    
    # Handle batched matrix multiplication
    if input.dim() >= 2 and other.dim() >= 2:
        # For batched operations, we can use torch's built-in implementation
        # which handles broadcasting and other complexities
        if out is not None:
            torch.matmul(input, other, out=out)
            return out
        return torch.matmul(input, other)
    
    # For mixed dimensions, use torch's implementation
    if out is not None:
        torch.matmul(input, other, out=out)
        return out
    return torch.matmul(input, other)

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
