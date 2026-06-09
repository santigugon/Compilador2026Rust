import torch
import triton
import triton.language as tl

def matmul(input, other, *, out=None):
    # Handle the case where one of the tensors is 1D
    if input.dim() == 1 and other.dim() == 1:
        # Dot product
        result = torch.dot(input, other)
        if out is not None:
            out.copy_(result)
            return out
        return result
    
    # For 2D tensors, use Triton kernel
    if input.dim() == 2 and other.dim() == 2:
        return _matmul_2d(input, other, out)
    
    # For higher dimensional tensors, use torch's matmul
    result = torch.matmul(input, other)
    if out is not None:
        out.copy_(result)
        return out
    return result

@triton.jit
def _matmul_2d_kernel(input_ptr, other_ptr, out_ptr, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr, stride_input_m: tl.constexpr, stride_input_k: tl.constexpr, stride_other_k: tl.constexpr, stride_other_n: tl.constexpr, stride_out_m: tl.constexpr, stride_out_n: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Get the program ID
    pid = tl.program_id(0)
    # Each program handles one block of the output matrix
    # Compute the block offsets
    m_block = (pid * BLOCK_M) // n
    n_block = (pid * BLOCK_M) % n
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over the K dimension
    for k in range(0, k, BLOCK_K):
        # Load input block
        input_block = tl.load(input_ptr + (m_block * stride_input_m + k * stride_input_k) + tl.arange(0, BLOCK_M)[:, None] * stride_input_m + tl.arange(0, BLOCK_K)[None, :] * stride_input_k)
        # Load other block
        other_block = tl.load(other_ptr + (k * stride_other_k + n_block * stride_other_n) + tl.arange(0, BLOCK_K)[:, None] * stride_other_k + tl.arange(0, BLOCK_N)[None, :] * stride_other_n)
        # Perform matrix multiplication
        acc += tl.dot(input_block, other_block)
    
    # Store the result
    out_block = acc
    tl.store(out_ptr + (m_block * stride_out_m + n_block * stride_out_n) + tl.arange(0, BLOCK_M)[:, None] * stride_out_m + tl.arange(0, BLOCK_N)[None, :] * stride_out_n, out_block)


def _matmul_2d(input, other, out=None):
    # Get dimensions
    m, k = input.shape
    k2, n = other.shape
    
    # Check for compatibility
    if k != k2:
        raise ValueError("Incompatible dimensions for matrix multiplication")
    
    # Allocate output tensor
    if out is None:
        out = torch.empty(m, n, dtype=input.dtype, device=input.device)
    else:
        if out.shape != (m, n):
            raise ValueError("Output tensor has incorrect shape")
    
    # Define block size
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    # Calculate grid size
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
    
    # Launch kernel
    _matmul_2d_kernel[grid](
        input, other, out,
        m, n, k,
        input.stride(0), input.stride(1),
        other.stride(0), other.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return out