import torch
import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub_kernel(
    input_ptr, vec_ptr, other_ptr, output_ptr,
    n, m,
    alpha,
    BLOCK_SIZE_M=32,
    BLOCK_SIZE_N=32
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute the matrix-vector multiplication
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for i in range(0, m, BLOCK_SIZE_N):
        # Load input matrix
        input_tile = tl.load(input_ptr + pid_m * m + i + tl.arange(0, BLOCK_SIZE_N), mask=i + tl.arange(0, BLOCK_SIZE_N) < m)
        # Load vector
        vec_tile = tl.load(vec_ptr + i + tl.arange(0, BLOCK_SIZE_N), mask=i + tl.arange(0, BLOCK_SIZE_N) < m)
        # Accumulate
        acc += input_tile * vec_tile
    
    # Compute sigmoid
    sigmoid_out = tl.sigmoid(acc)
    
    # Load other tensor
    other_val = tl.load(other_ptr + pid_m)
    
    # Apply subtraction with alpha scaling
    result = sigmoid_out - alpha * other_val
    
    # Store result
    tl.store(output_ptr + pid_m, result)

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    assert input.dim() == 2
    assert vec.dim() == 1
    assert input.size(1) == vec.size(0)
    
    n, m = input.shape
    
    if out is None:
        out = torch.empty(n, dtype=torch.float32, device=input.device)
    
    # Ensure other is a tensor
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, dtype=torch.float32, device=input.device)
    
    if other.dim() == 0:
        other = other.expand(n)
    else:
        assert other.size(0) == n
    
    # Launch kernel
    grid = (triton.cdiv(n, 32), triton.cdiv(m, 32))
    fused_mv_sigmoid_sub_kernel[grid](
        input_ptr=input.data_ptr(),
        vec_ptr=vec.data_ptr(),
        other_ptr=other.data_ptr(),
        output_ptr=out.data_ptr(),
        n=n,
        m=m,
        alpha=alpha
    )
    
    return out
