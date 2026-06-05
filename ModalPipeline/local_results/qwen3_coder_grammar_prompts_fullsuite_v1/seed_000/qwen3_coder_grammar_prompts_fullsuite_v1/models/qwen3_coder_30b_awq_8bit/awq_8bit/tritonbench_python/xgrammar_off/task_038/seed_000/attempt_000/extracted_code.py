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
    rand = tl.random.rand(0, 1)
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
    # Matrix multiplication 1: Z_1 = input @ weight1
    Z_1 = torch.matmul(input, weight1)
    
    # Softmax: Z_2 = softmax(Z_1)
    Z_2 = torch.empty_like(Z_1)
    n = Z_1.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _softmax_kernel[grid](Z_1, Z_2, n, BLOCK=block)
    
    # Dropout: Z_3 = dropout(Z_2)
    Z_3 = torch.empty_like(Z_2)
    _dropout_kernel[grid](Z_2, Z_3, n, dropout_p, BLOCK=block)
    
    # Matrix multiplication 2: Z_4 = Z_3 @ weight2
    Z_4 = torch.matmul(Z_3, weight2)
    
    # Add residual: Z_5 = Z_4 + residual
    Z_5 = Z_4 + residual
    
    # Layer normalization: output = layer_norm(Z_5)
    output = torch.empty_like(Z_5)
    n = Z_5.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # For layer norm, we need to compute mean and variance per row
    # This is a simplified version that assumes the last dimension is the feature dimension
    # and we're normalizing across that dimension
    if len(Z_5.shape) > 1:
        batch_size = Z_5.shape[:-1]
        feature_dim = Z_5.shape[-1]
        # Flatten for processing
        Z_5_flat = Z_5.view(-1, feature_dim)
        output_flat = output.view(-1, feature_dim)
        
        # Compute mean and variance for each row
        mean = torch.empty(Z_5_flat.shape[0], dtype=torch.float32, device=Z_5.device)
        var = torch.empty(Z_5_flat.shape[0], dtype=torch.float32, device=Z_5.device)
        
        # For simplicity, we'll use PyTorch's layer norm for the actual computation
        # since it's more complex to implement in Triton with proper broadcasting
        for i in range(Z_5_flat.shape[0]):
            output_flat[i] = torch.nn.functional.layer_norm(
                Z_5_flat[i:i+1], 
                normalized_shape=(feature_dim,), 
                eps=eps
            ).squeeze(0)
    else:
        # For 1D case, use PyTorch's layer norm directly
        output = torch.nn.functional.layer_norm(Z_5, normalized_shape=(Z_5.shape[-1],), eps=eps)
    
    # If out is provided, copy result to it
    if out is not None:
        out.copy_(output)
        return out
    
    return output
