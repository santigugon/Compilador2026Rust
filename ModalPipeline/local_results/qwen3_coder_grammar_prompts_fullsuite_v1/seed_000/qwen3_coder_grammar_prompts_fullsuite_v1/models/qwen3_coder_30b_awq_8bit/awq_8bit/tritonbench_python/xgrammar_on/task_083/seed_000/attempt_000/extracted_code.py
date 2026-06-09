import torch
import triton
import triton.language as tl

def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if approximate == 'tanh':
        y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    else:
        y = 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr + offsets, y, mask=mask)

def _dropout_kernel(x_ptr, out_ptr, mask_ptr, n: tl.constexpr, p: tl.constexpr, training: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    if training:
        # Generate random mask
        rand = tl.random.rand(0, BLOCK)  # Using a fixed seed for reproducibility
        keep_mask = rand > p
        tl.store(mask_ptr + offsets, keep_mask, mask=mask)
        y = x * keep_mask / (1.0 - p)
    else:
        y = x
        tl.store(mask_ptr + offsets, tl.full(BLOCK, True, tl.bool), mask=mask)
    tl.store(out_ptr + offsets, y, mask=mask)

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    # Check input shapes
    assert input1.dim() == 3 and input2.dim() == 3, "Both inputs must be 3D tensors"
    assert input1.size(0) == input2.size(0), "Batch dimensions must match"
    assert input1.size(2) == input2.size(1), "Matrix dimensions must be compatible for multiplication"
    
    B, N, M = input1.shape
    _, _, P = input2.shape
    
    # Compute batch matrix multiplication
    bmm_out = torch.bmm(input1, input2)
    
    # Prepare output tensor
    if out is not None:
        output = out
    else:
        output = torch.empty_like(bmm_out)
    
    # Apply dropout
    if training:
        dropout_mask = torch.empty_like(bmm_out, dtype=torch.bool)
        n = bmm_out.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        _dropout_kernel[grid](bmm_out, output, dropout_mask, n, p, training, BLOCK=block)
    else:
        output = bmm_out
        
    # Apply GELU
    n = output.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _gelu_kernel[grid](output, output, n, approximate, BLOCK=block)
    
    return output