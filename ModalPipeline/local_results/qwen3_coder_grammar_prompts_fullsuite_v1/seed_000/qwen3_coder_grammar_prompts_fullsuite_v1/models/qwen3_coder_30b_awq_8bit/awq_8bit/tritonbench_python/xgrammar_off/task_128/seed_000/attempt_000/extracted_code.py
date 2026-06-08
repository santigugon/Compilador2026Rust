import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(x_ptr, rms_ptr, weight_ptr, out_ptr, n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    
    # Compute RMS
    x2 = x * x
    mean = tl.sum(x2, axis=0) / n
    rms = tl.sqrt(mean + eps)
    
    # Normalize and scale
    normalized = x / rms
    out = normalized * weight
    
    tl.store(rms_ptr + pid, rms)
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, approximate: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 'tanh':
        # GELU approximation using tanh
        y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    else:
        # Standard GELU
        y = 0.5 * x * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _dropout_kernel(x_ptr, out_ptr, mask_ptr, n: tl.constexpr, dropout_p: tl.constexpr, training: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if training:
        # Generate random mask
        rand = tl.rand(tl.program_id(0), offsets[0])  # Simple random generation
        keep_mask = rand > dropout_p
        y = x * keep_mask / (1.0 - dropout_p)
    else:
        y = x
    
    tl.store(out_ptr + offsets, y, mask=mask)
    if mask_ptr is not None:
        tl.store(mask_ptr + offsets, keep_mask, mask=mask)

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    # Perform batch matrix multiplication
    bmm_out = torch.bmm(input1, input2)
    
    # Prepare for RMS normalization
    batch_size, seq_len, output_size = bmm_out.shape
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    
    # Flatten for normalization
    flat_input = bmm_out.view(-1, output_size)
    flat_output = torch.empty_like(flat_input)
    
    # RMS normalization
    weight = torch.ones(output_size, dtype=flat_input.dtype, device=flat_input.device)
    rms_values = torch.empty(batch_size * seq_len, dtype=torch.float32, device=flat_input.device)
    
    n = flat_input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Compute RMS normalization
    _rmsnorm_kernel[grid](flat_input, rms_values, weight, flat_output, n, eps, BLOCK=block)
    
    # Reshape back
    norm_out = flat_output.view(batch_size, seq_len, output_size)
    
    # Apply GELU
    gelu_out = torch.empty_like(norm_out)
    n = norm_out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    approx = 'tanh' if approximate == 'tanh' else 'none'
    _gelu_kernel[grid](norm_out, gelu_out, n, approx, BLOCK=block)
    
    # Apply dropout
    if training and dropout_p > 0:
        dropout_out = torch.empty_like(gelu_out)
        n = gelu_out.numel()
        block = 256
        grid = (triton.cdiv(n, block),)
        
        # Create dropout mask
        dropout_mask = torch.empty_like(gelu_out, dtype=torch.bool)
        _dropout_kernel[grid](gelu_out, dropout_out, dropout_mask, n, dropout_p, training, BLOCK=block)
        final_out = dropout_out
    else:
        final_out = gelu_out
    
    if out is not None:
        out.copy_(final_out)
        return out
    return final_out
