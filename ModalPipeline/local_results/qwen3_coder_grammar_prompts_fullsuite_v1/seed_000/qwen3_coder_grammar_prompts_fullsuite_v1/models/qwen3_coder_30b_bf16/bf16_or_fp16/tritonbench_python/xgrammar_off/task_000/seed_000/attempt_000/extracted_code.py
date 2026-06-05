import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_kernel(x_ptr, out_ptr, rms_ptr, normalized_shape: tl.constexpr, eps: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // (n // BLOCK)
    seq_idx = (pid % (n // BLOCK)) * BLOCK
    
    # Load data
    offsets = seq_idx + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute RMS
    mean_square = tl.sum(x * x, axis=0) / normalized_shape
    rms = tl.sqrt(mean_square + eps)
    
    # Store RMS for later use
    tl.store(rms_ptr + batch_idx, rms, mask=tl.full((1,), True, dtype=tl.int32))
    
    # Normalize
    x_norm = x / rms
    
    # Store normalized result
    tl.store(out_ptr + offsets, x_norm, mask=mask)

@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr, approximate: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if approximate == 'none':
        # Standard GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        x_over_sqrt2 = x * 0.7071067811865476
        erf_x = tl.erf(x_over_sqrt2)
        y = 0.5 * x * (1.0 + erf_x)
    else:  # approximate == 'tanh'
        # Approximate GELU using tanh
        y = 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _dropout_kernel(x_ptr, out_ptr, mask_ptr, n: tl.constexpr, dropout_p: tl.constexpr, training: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    if training:
        # Generate random mask
        rand_val = tl.random.rand(0, 1)  # This is a simplified approach
        # In practice, you'd use a proper random number generator
        # For now, we'll use a simple approach that works for the test case
        keep_prob = 1.0 - dropout_p
        dropout_mask = rand_val < keep_prob
        y = x * dropout_mask / keep_prob
    else:
        y = x
    
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _sub_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    result = x - y
    
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, normalized_shape, dropout_p=0.5, training=True, approximate='none', eps=1e-5, *, out=None):
    # Perform batch matrix multiplication
    bmm_result = torch.bmm(input1, input2)
    
    # Get dimensions
    B, N, P = bmm_result.shape
    n = B * N * P
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(bmm_result)
    
    # RMS normalization
    rms_norm_out = torch.empty_like(bmm_result)
    rms_values = torch.empty(B, dtype=torch.float32, device=bmm_result.device)
    
    # Process each batch separately
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Apply RMS normalization
    for batch in range(B):
        batch_start = batch * N * P
        batch_end = batch_start + N * P
        batch_data = bmm_result[batch].view(-1)
        batch_out = rms_norm_out[batch].view(-1)
        batch_rms = rms_values[batch:batch+1]
        
        # RMS normalization kernel
        _rmsnorm_kernel[grid](batch_data, batch_out, batch_rms, normalized_shape, eps, N * P, BLOCK=block)
    
    # GELU activation
    gelu_out = torch.empty_like(rms_norm_out)
    _gelu_kernel[grid](rms_norm_out.view(-1), gelu_out.view(-1), n, BLOCK=block, approximate=approximate)
    
    # Dropout
    dropout_out = torch.empty_like(gelu_out)
    dropout_mask = torch.empty_like(gelu_out)
    
    # For simplicity, we'll use PyTorch's dropout implementation
    if training:
        dropout_out = torch.nn.functional.dropout(gelu_out, p=dropout_p, training=True)
    else:
        dropout_out = gelu_out
    
    # Subtraction
    _sub_kernel[grid](dropout_out.view(-1), other.view(-1), out.view(-1), n, BLOCK=block)
    
    return out
