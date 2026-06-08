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
    # Handle 1D case (dot product)
    if input.dim() == 1 and other.dim() == 1:
        if out is not None:
            raise ValueError("1D dot product does not support out parameter")
        return torch.dot(input, other)
    
    # Handle 2D case (matrix multiplication)
    if input.dim() == 2 and other.dim() == 2:
        if out is None:
            out = torch.empty(input.size(0), other.size(1), dtype=input.dtype, device=input.device)
        else:
            assert out.shape == (input.size(0), other.size(1)), "Output tensor has incorrect shape"
        
        # Determine block size and grid size
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        GROUP_M = 8
        
        # Calculate grid size
        grid = (triton.cdiv(input.size(0), BLOCK_M) * triton.cdiv(other.size(1), BLOCK_N),)
        
        # Launch kernel
        _matmul_kernel[grid](
            input, other, out,
            input.size(0), other.size(1), input.size(1),
            input.stride(0), input.stride(1),
            other.stride(0), other.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M
        )
        return out
    
    # Handle batched matrix multiplication
    if input.dim() > 2 or other.dim() > 2:
        # Use PyTorch's built-in matmul for higher dimensional cases
        if out is None:
            return torch.matmul(input, other)
        else:
            torch.matmul(input, other, out=out)
            return out
    
    # Handle mixed 1D and 2D case (matrix-vector product)
    if input.dim() == 2 and other.dim() == 1:
        if out is None:
            out = torch.empty(input.size(0), dtype=input.dtype, device=input.device)
        else:
            assert out.shape == (input.size(0),), "Output tensor has incorrect shape"
        
        # Use PyTorch's built-in matmul for matrix-vector product
        torch.matmul(input, other, out=out)
        return out
    
    if input.dim() == 1 and other.dim() == 2:
        if out is None:
            out = torch.empty(other.size(1), dtype=other.dtype, device=other.device)
        else:
            assert out.shape == (other.size(1),), "Output tensor has incorrect shape"
        
        # Use PyTorch's built-in matmul for vector-matrix product
        torch.matmul(other, input, out=out)
        return out
    
    # Fallback to PyTorch for unsupported cases
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
