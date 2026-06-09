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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    # Create block pointers for A, B, C
    pid = tl.program_id(0)
    pid_m = pid // GROUP_M
    pid_n = pid % GROUP_M
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A and B with appropriate masks
        a = tl.load(A_ptr + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_am +
                    (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak,
                    mask=(k + tl.arange(0, BLOCK_K))[None, :] < K)
        
        b = tl.load(B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * stride_bk +
                    (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_bn,
                    mask=(k + tl.arange(0, BLOCK_K))[:, None] < K)
        
        # Perform matrix multiplication
        acc += tl.dot(a, b)
    
    # Store result
    c = acc.to(tl.float32)
    tl.store(C_ptr + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_cm +
             (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_cn,
             c)

def matmul(input, other, *, out=None):
    # Handle the case where one of the tensors is 1D
    if input.dim() == 1 and other.dim() == 1:
        # Dot product case
        if out is not None:
            raise ValueError("out parameter not supported for 1D dot product")
        return torch.dot(input, other)
    
    # For 2D or higher dimensional tensors, use Triton kernel
    if input.dim() == 2 and other.dim() == 2:
        # Standard matrix multiplication
        M, K = input.shape
        K, N = other.shape
        if out is None:
            out = torch.empty(M, N, dtype=input.dtype, device=input.device)
        else:
            assert out.shape == (M, N), "Output tensor shape mismatch"
        
        # Define block sizes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        GROUP_M = 8
        
        # Calculate grid size
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        
        # Launch kernel
        _matmul_kernel[grid](
            input, other, out,
            M, N, K,
            input.stride(0), input.stride(1),
            other.stride(0), other.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M
        )
        return out
    
    # For higher dimensional tensors, use torch's built-in matmul
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
