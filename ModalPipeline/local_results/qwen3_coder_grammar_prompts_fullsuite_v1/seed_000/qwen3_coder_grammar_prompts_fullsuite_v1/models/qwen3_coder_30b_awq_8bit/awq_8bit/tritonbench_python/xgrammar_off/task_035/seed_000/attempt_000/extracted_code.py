import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, out_ptr,
    n: tl.constexpr, m: tl.constexpr,
    alpha: tl.constexpr,
    input_stride_0: tl.constexpr, input_stride_1: tl.constexpr,
    vec_stride: tl.constexpr,
    other_stride: tl.constexpr,
    out_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid
    
    if row >= n:
        return
    
    # Initialize accumulator for matrix-vector multiplication
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Perform matrix-vector multiplication for this row
    for i in range(0, m, BLOCK_SIZE):
        vec_offsets = i + tl.arange(0, BLOCK_SIZE)
        input_offsets = row * input_stride_0 + vec_offsets * input_stride_1
        other_offsets = vec_offsets * other_stride
        
        # Load vector elements with mask
        vec_vals = tl.load(vec_ptr + vec_offsets, mask=vec_offsets < m, other=0.0)
        
        # Load input elements with mask
        input_vals = tl.load(input_ptr + input_offsets, mask=vec_offsets < m, other=0.0)
        
        # Accumulate dot product
        acc += tl.sum(input_vals * vec_vals)
    
    # Compute sigmoid of the result
    sigmoid_val = 1.0 / (1.0 + tl.exp(-acc))
    
    # Load other value (could be scalar or tensor)
    if other_ptr is not None:
        other_val = tl.load(other_ptr + row * other_stride, mask=row < n, other=0.0)
    else:
        other_val = 0.0
    
    # Apply alpha scaling and subtraction
    result = sigmoid_val - alpha * other_val
    
    # Store result
    tl.store(out_ptr + row * out_stride, result, mask=row < n)

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    # Validate input shapes
    n, m = input.shape
    assert vec.shape == (m,), f"Vector shape {vec.shape} does not match expected (m,) where m={m}"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty((n,), dtype=input.dtype, device=input.device)
    else:
        assert out.shape == (n,), f"Output shape {out.shape} does not match expected (n,)"
        assert out.dtype == input.dtype, f"Output dtype {out.dtype} does not match input dtype {input.dtype}"
        assert out.device == input.device, f"Output device {out.device} does not match input device {input.device}"
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=input.dtype, device=input.device)
    
    # Ensure other is broadcastable to (n,)
    if other.shape == ():
        other = other.expand((n,))
    elif other.shape != (n,):
        raise ValueError(f"Other shape {other.shape} is not broadcastable to (n,)")
    
    # Launch kernel
    block_size = 256
    grid_size = triton.cdiv(n, block_size)
    
    # Get strides
    input_stride_0, input_stride_1 = input.stride()
    vec_stride = vec.stride(0) if vec.numel() > 0 else 1
    other_stride = other.stride(0) if other.numel() > 0 else 1
    out_stride = out.stride(0) if out.numel() > 0 else 1
    
    # Handle case where other is a scalar (no pointer needed)
    other_ptr = other.data_ptr() if other.numel() > 0 else None
    
    _fused_mv_sigmoid_sub_kernel[grid_size](
        input.data_ptr(), vec.data_ptr(), other_ptr, out.data_ptr(),
        n, m, alpha,
        input_stride_0, input_stride_1,
        vec_stride,
        other_stride,
        out_stride,
        BLOCK_SIZE=block_size
    )
    
    return out
