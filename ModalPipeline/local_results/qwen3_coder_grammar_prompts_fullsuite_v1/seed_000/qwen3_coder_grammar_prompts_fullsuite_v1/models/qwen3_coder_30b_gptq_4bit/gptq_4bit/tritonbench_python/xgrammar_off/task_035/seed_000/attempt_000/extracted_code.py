import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, out_ptr,
    n: tl.constexpr, m: tl.constexpr,
    input_stride_0: tl.constexpr, input_stride_1: tl.constexpr,
    vec_stride: tl.constexpr,
    other_stride: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    pid_m = pid % (n // BLOCK_M)
    pid_n = pid // (n // BLOCK_M)
    
    # Load input matrix row
    input_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = input_offsets < n
    
    # Load vector
    vec_offsets = tl.arange(0, BLOCK_N)
    mask_n = vec_offsets < m
    
    # Compute matrix-vector multiplication
    accumulator = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for i in range(0, m, BLOCK_N):
        # Load input row and vector
        input_row = tl.load(input_ptr + input_offsets * input_stride_0 + i * input_stride_1, mask=mask_m & (vec_offsets + i < m), other=0.0)
        vec = tl.load(vec_ptr + vec_offsets + i, mask=mask_n & (vec_offsets + i < m), other=0.0)
        accumulator += input_row * vec
    
    # Compute sigmoid
    sigmoid_out = 1.0 / (1.0 + tl.exp(-accumulator))
    
    # Load other tensor
    other = tl.load(other_ptr + pid_m * other_stride, mask=mask_m, other=0.0)
    
    # Apply alpha scaling and subtraction
    result = sigmoid_out - other * 1.0  # alpha is always 1
    
    # Store result
    tl.store(out_ptr + input_offsets, result, mask=mask_m)

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    # Validate inputs
    assert input.dim() == 2, "input must be a 2D tensor"
    assert vec.dim() == 1, "vec must be a 1D tensor"
    assert input.size(1) == vec.size(0), "input and vec dimensions must match"
    
    n, m = input.shape
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other has the right shape for broadcasting
    if other.dim() == 0:
        other = other.expand(n)
    elif other.dim() == 1:
        if other.size(0) == 1:
            other = other.expand(n)
        else:
            assert other.size(0) == n, "other must be scalar or have size n"
    
    # Create output tensor
    if out is None:
        out = torch.empty(n, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == (n,), "out must have shape (n,)"
        assert out.dtype == input.dtype, "out must have the same dtype as input"
        assert out.device == input.device, "out must be on the same device as input"
    
    # Set up grid and block sizes
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(n, BLOCK_M),)
    
    # Launch kernel
    _fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out,
        n, m,
        input.stride(0), input.stride(1),
        vec.stride(0),
        other.stride(0),
        BLOCK_M, BLOCK_N
    )
    
    return out
