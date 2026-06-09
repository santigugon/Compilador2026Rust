import torch
import triton
import triton.language as tl

@triton.jit
def _fused_mv_sigmoid_sub_kernel(A_ptr, v_ptr, other_ptr, out_ptr, n: tl.constexpr, m: tl.constexpr, alpha: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Each program processes one row of the output
    row = pid
    if row >= n:
        return
    
    # Compute A[row, :] @ v
    acc = tl.zeros((1,), dtype=tl.float32)
    for i in range(0, m, BLOCK):
        v_offsets = i + tl.arange(0, BLOCK)
        a_offsets = row * m + v_offsets
        
        mask_v = v_offsets < m
        mask_a = a_offsets < (row + 1) * m
        
        v_vals = tl.load(v_ptr + v_offsets, mask=mask_v, other=0.0)
        a_vals = tl.load(A_ptr + a_offsets, mask=mask_a, other=0.0)
        
        acc += tl.sum(a_vals * v_vals)
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-acc))
    
    # Load other (scalar or tensor)
    other_val = tl.load(other_ptr)
    result = sigmoid_val - alpha * other_val
    
    # Store result
    tl.store(out_ptr + row, result)


def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    assert input.dim() == 2, "input must be a 2D tensor"
    assert vec.dim() == 1, "vec must be a 1D tensor"
    assert input.size(1) == vec.size(0), "input and vec dimensions must match"
    
    n, m = input.shape
    
    if out is None:
        out = torch.empty(n, dtype=torch.float32, device=input.device)
    
    # Handle scalar other
    if not torch.is_tensor(other):
        other = torch.tensor(other, dtype=torch.float32, device=input.device)
    
    # Ensure other is a scalar tensor
    assert other.numel() == 1, "other must be a scalar or a 1-element tensor"
    
    block = 256
    grid = (n,)
    
    _fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out, n, m, alpha, BLOCK=block
    )
    
    return out