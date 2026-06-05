import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    # Subtract max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    # Compute softmax
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _layer_norm_kernel(x_ptr, weight_ptr, out_ptr, mean_ptr, n: tl.constexpr, normalized_shape: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets % normalized_shape, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / normalized_shape
    tl.store(mean_ptr + pid, mean, mask=pid < tl.cdiv(n, BLOCK))
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / normalized_shape
    
    # Layer norm
    x_norm = x_centered / tl.sqrt(var + eps)
    y = x_norm * weight
    
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _cross_entropy_kernel(logits_ptr, targets_ptr, out_ptr, n: tl.constexpr, C: tl.constexpr, ignore_index: tl.constexpr, label_smoothing: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    logits = tl.load(logits_ptr + offsets * C, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cross entropy with label smoothing
    # For simplicity, we'll compute it in a basic way
    # In practice, this would be more complex for full label smoothing
    
    # Find max for numerical stability
    logits_max = tl.max(logits, axis=0)
    logits_centered = logits - logits_max
    logits_exp = tl.exp(logits_centered)
    logits_sum = tl.sum(logits_exp, axis=0)
    
    # Compute log probabilities
    log_probs = logits_centered - tl.log(logits_sum)
    
    # Compute loss
    loss = 0.0
    if label_smoothing > 0:
        # Apply label smoothing
        smooth_loss = -tl.sum(log_probs, axis=0) / C
        # This is a simplified version - full implementation would be more complex
        loss = smooth_loss
    else:
        # Standard cross entropy
        target_indices = targets.to(tl.int32)
        loss = -log_probs[target_indices]
    
    # Apply ignore_index
    ignore_mask = targets != ignore_index
    loss = tl.where(ignore_mask, loss, 0.0)
    
    tl.store(out_ptr + offsets, loss, mask=mask)

def fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None):
    # Handle different input shapes
    if logits.dim() == 2:
        N, C = logits.shape
    else:
        N, C = logits.shape[0], logits.shape[1]
        # Flatten for processing
        logits = logits.view(N, C)
        targets = targets.view(N)
    
    # Compute softmax
    softmax_out = torch.empty_like(logits)
    block = 256
    grid = (triton.cdiv(N * C, block),)
    _softmax_kernel[grid](logits, softmax_out, N * C, BLOCK=block)
    
    # Compute cross entropy loss
    ce_loss = torch.empty(N, dtype=logits.dtype, device=logits.device)
    _cross_entropy_kernel[grid](logits, targets, ce_loss, N, C, ignore_index, label_smoothing, BLOCK=block)
    
    # Apply reduction
    if reduction == 'mean':
        loss = ce_loss.mean()
    elif reduction == 'sum':
        loss = ce_loss.sum()
    else:  # 'none'
        loss = ce_loss
    
    # Apply layer normalization
    if weight is None:
        weight = torch.ones(normalized_shape, dtype=logits.dtype, device=logits.device)
    
    if out is None:
        out_tensor = torch.empty_like(softmax_out)
    else:
        out_tensor = out
    
    # Layer norm computation
    mean = torch.empty(triton.cdiv(N * C, block), dtype=torch.float32, device=logits.device)
    _layer_norm_kernel[grid](softmax_out, weight, out_tensor, mean, N * C, normalized_shape, eps, BLOCK=block)
    
    return loss, out_tensor
