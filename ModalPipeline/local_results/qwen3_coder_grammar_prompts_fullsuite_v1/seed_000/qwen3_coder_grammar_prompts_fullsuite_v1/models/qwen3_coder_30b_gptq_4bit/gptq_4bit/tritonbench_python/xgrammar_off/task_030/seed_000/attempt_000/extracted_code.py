import torch
import triton
import triton.language as tl
import math

@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x - tl.max(x, axis=0)
    y = tl.exp(x)
    y = y / tl.sum(y, axis=0)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _layer_norm_kernel(x_ptr, weight_ptr, out_ptr, n: tl.constexpr, normalized_shape: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0.0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and variance
    mean = tl.sum(x, axis=0) / normalized_shape
    var = tl.sum((x - mean) * (x - mean), axis=0) / normalized_shape
    
    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)
    
    # Apply weight
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        x_norm = x_norm * weight
    
    tl.store(out_ptr + offsets, x_norm, mask=mask)

@triton.jit
def _cross_entropy_kernel(logits_ptr, targets_ptr, out_ptr, n: tl.constexpr, C: tl.constexpr, ignore_index: tl.constexpr, label_smoothing: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cross entropy loss
    loss = 0.0
    for i in range(C):
        if targets[i] != ignore_index:
            loss += logits[i] * (1.0 - label_smoothing) + label_smoothing / C
    
    tl.store(out_ptr + offsets, loss, mask=mask)

def fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None):
    # Flatten logits and targets to 2D
    original_shape = logits.shape
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    
    # Flatten logits and targets
    logits_flat = logits.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)
    
    # Compute softmax
    softmax_out = torch.empty_like(logits_flat)
    n = logits_flat.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _softmax_kernel[grid](logits_flat, softmax_out, n, BLOCK=block)
    
    # Apply layer normalization
    if weight is not None:
        weight_tensor = weight
    else:
        weight_tensor = torch.ones_like(softmax_out)
    
    if out is not None:
        normalized_out = out
    else:
        normalized_out = torch.empty_like(softmax_out)
    
    n = softmax_out.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _layer_norm_kernel[grid](softmax_out, weight_tensor, normalized_out, n, num_classes, eps, BLOCK=block)
    
    # Compute cross entropy loss
    loss_out = torch.empty(batch_size, dtype=torch.float32)
    n = batch_size
    block = 256
    grid = (triton.cdiv(n, block),)
    _cross_entropy_kernel[grid](logits_flat, targets_flat, loss_out, n, num_classes, ignore_index, label_smoothing, BLOCK=block)
    
    # Apply reduction
    if reduction == 'mean':
        loss = loss_out.mean()
    elif reduction == 'sum':
        loss = loss_out.sum()
    else:
        loss = loss_out
    
    return (normalized_out, loss)
