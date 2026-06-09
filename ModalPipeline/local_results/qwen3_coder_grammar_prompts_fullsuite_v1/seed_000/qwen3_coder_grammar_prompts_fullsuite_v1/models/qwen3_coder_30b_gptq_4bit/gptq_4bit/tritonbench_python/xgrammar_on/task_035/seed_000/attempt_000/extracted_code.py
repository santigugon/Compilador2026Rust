import torch
import triton
import triton.language as tl

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    # Ensure input is a tensor
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    if not torch.is_tensor(vec):
        vec = torch.tensor(vec)
    if not torch.is_tensor(other):
        other = torch.tensor(other)

    # Get dimensions
    n, m = input.shape
    assert vec.shape[0] == m, "Vector dimension must match matrix columns"

    # Prepare output tensor
    if out is None:
        out = torch.empty(n, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == (n,), "Output tensor must have shape (n,)"

    # Define block size
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)

    # Launch kernel
    _fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out,
        n, m,
        input.stride(0), input.stride(1),
        vec.stride(0),
        other.stride(0) if other.dim() > 0 else 0,
        BLOCK=BLOCK,
        alpha=alpha
    )

    return out

@triton.jit

# Triton kernel for fused matrix-vector multiplication, sigmoid, and subtraction

# Kernel parameters:
# - input: (n, m) matrix
# - vec: (m,) vector
# - other: (n,) tensor or scalar
# - out: (n,) output tensor
# - n, m: dimensions
# - input_stride_0, input_stride_1: strides for input matrix
# - vec_stride_0: stride for vector
# - other_stride_0: stride for other tensor
# - alpha: scalar multiplier for other

def _fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, out_ptr,
    n: tl.constexpr, m: tl.constexpr,
    input_stride_0: tl.constexpr, input_stride_1: tl.constexpr,
    vec_stride_0: tl.constexpr,
    other_stride_0: tl.constexpr,
    BLOCK: tl.constexpr,
    alpha: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets for the block
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Create mask for valid elements
    mask = offsets < n
    
    # Initialize output
    out = tl.zeros((BLOCK,), dtype=tl.float32)
    
    # Perform matrix-vector multiplication for each row
    for i in range(m):
        # Load input element
        input_val = tl.load(input_ptr + offsets * input_stride_0 + i * input_stride_1, mask=mask, other=0.0)
        # Load vector element
        vec_val = tl.load(vec_ptr + i * vec_stride_0, other=0.0)
        # Accumulate
        out += input_val * vec_val
    
    # Apply sigmoid
    out = 1.0 / (1.0 + tl.exp(-out))
    
    # Load other tensor
    other_val = tl.load(other_ptr + offsets * other_stride_0, mask=mask, other=0.0)
    
    # Apply subtraction with alpha scaling
    out = out - alpha * other_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)