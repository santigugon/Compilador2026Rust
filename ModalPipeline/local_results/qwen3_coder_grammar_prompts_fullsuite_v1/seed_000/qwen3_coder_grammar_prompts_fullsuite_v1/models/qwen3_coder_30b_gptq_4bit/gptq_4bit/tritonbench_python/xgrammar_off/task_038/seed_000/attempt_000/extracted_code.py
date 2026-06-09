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
    y = tl.exp(x)
    y = y / tl.sum(y, axis=0)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _dropout_kernel(x_ptr, out_ptr, n: tl.constexpr, dropout_p: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    rand = tl.random.rand(0, 1)
    keep_prob = 1.0 - dropout_p
    mask = rand > keep_prob
    y = x * mask / keep_prob
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _layer_norm_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, n: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n
    var = tl.sum((x - mean) ** 2, axis=0) / n
    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * tl.load(weight_ptr + offsets, mask=mask, other=0.0) + tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    # input shape: (*, N, D_in)
    # weight1 shape: (D_in, D_k)
    # weight2 shape: (D_k, D_out)
    # residual shape: broadcastable to (*, N, D_out)
    
    batch_shape = input.shape[:-2]
    N, D_in = input.shape[-2], weight1.shape[0]
    D_k, D_out = weight1.shape[1], weight2.shape[1]
    
    # First matrix multiplication: Z1 = input @ weight1
    Z1 = torch.empty(*batch_shape, N, D_k, dtype=input.dtype, device=input.device)
    Z1 = torch.mm(input.view(-1, D_in), weight1).view(*batch_shape, N, D_k)
    
    # Softmax
    Z2 = torch.empty_like(Z1)
    n = Z1.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _softmax_kernel[grid](Z1, Z2, n, BLOCK=block)
    
    # Dropout
    Z3 = torch.empty_like(Z2)
    _dropout_kernel[grid](Z2, Z3, n, dropout_p, BLOCK=block)
    
    # Second matrix multiplication: Z4 = Z3 @ weight2
    Z4 = torch.empty(*batch_shape, N, D_out, dtype=input.dtype, device=input.device)
    Z4 = torch.mm(Z3.view(-1, D_k), weight2).view(*batch_shape, N, D_out)
    
    # Add residual
    Z5 = Z4 + residual
    
    # Layer normalization
    Z6 = torch.empty_like(Z5)
    n = Z5.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    
    # Create weight and bias for layer norm
    weight = torch.ones(D_out, dtype=input.dtype, device=input.device)
    bias = torch.zeros(D_out, dtype=input.dtype, device=input.device)
    
    _layer_norm_kernel[grid](Z5, weight, bias, Z6, n, eps, BLOCK=block)
    
    if out is not None:
        out.copy_(Z6)
        return out
    return Z6
