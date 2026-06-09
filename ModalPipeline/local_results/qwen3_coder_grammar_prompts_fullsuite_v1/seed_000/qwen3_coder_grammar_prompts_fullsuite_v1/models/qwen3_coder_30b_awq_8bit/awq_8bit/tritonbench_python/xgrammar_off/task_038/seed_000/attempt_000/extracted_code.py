import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    x = x - tl.max(x, axis=0)
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    softmax_x = exp_x / sum_exp_x
    tl.store(out_ptr + offsets, softmax_x, mask=mask)

@triton.jit
def _dropout_kernel(x_ptr, out_ptr, n: tl.constexpr, dropout_p: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    rand = tl.random.rand(0, 0)  # Use a fixed seed for reproducibility
    keep_mask = rand > dropout_p
    y = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, mean_ptr, var_ptr, 
                       n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n
    var = tl.sum((x - mean) ** 2, axis=0) / n
    x_norm = (x - mean) / tl.sqrt(var + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    y = x_norm * weight + bias
    tl.store(out_ptr + offsets, y, mask=mask)
    tl.store(mean_ptr + pid, mean, mask=pid < 1)
    tl.store(var_ptr + pid, var, mask=pid < 1)

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    # Compute Z1 = input @ weight1
    Z1 = torch.matmul(input, weight1)
    
    # Compute Z2 = softmax(Z1)
    Z2 = torch.empty_like(Z1)
    n = Z1.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _softmax_kernel[grid](Z1, Z2, n, BLOCK=block)
    
    # Compute Z3 = dropout(Z2)
    Z3 = torch.empty_like(Z2)
    _dropout_kernel[grid](Z2, Z3, n, dropout_p, BLOCK=block)
    
    # Compute Z4 = Z3 @ weight2
    Z4 = torch.matmul(Z3, weight2)
    
    # Add residual to Z4
    Z4 = Z4 + residual
    
    # Compute layer normalization
    # For layer norm, we need to compute mean and variance over the last dimension
    # Assuming the last dimension is D_out, we need to reshape for proper computation
    input_shape = Z4.shape
    D_out = input_shape[-1]
    Z4_flat = Z4.view(-1, D_out)
    
    # Compute mean and variance
    mean = Z4_flat.mean(dim=1, keepdim=True)
    var = Z4_flat.var(dim=1, keepdim=True, unbiased=False)
    
    # Layer normalization
    Z5 = (Z4_flat - mean) / torch.sqrt(var + eps)
    Z5 = Z5.view(input_shape)
    
    # Final output
    if out is not None:
        out.copy_(Z5)
        return out
    else:
        return Z5
