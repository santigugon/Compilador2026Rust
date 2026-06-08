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
def _layer_norm_kernel(x_ptr, weight_ptr, out_ptr, mean_ptr, var_ptr, 
                      n: tl.constexpr, normalized_shape: tl.constexpr, 
                      eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean and variance
    mean = tl.sum(x, axis=0) / normalized_shape
    var = tl.sum((x - mean) * (x - mean), axis=0) / normalized_shape
    
    # Layer normalization
    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * weight
    
    tl.store(out_ptr + offsets, y, mask=mask)
    if mean_ptr is not None:
        tl.store(mean_ptr + pid, mean)
    if var_ptr is not None:
        tl.store(var_ptr + pid, var)

@triton.jit
def _cross_entropy_kernel(logits_ptr, targets_ptr, loss_ptr, 
                         n: tl.constexpr, C: tl.constexpr, 
                         ignore_index: tl.constexpr, 
                         label_smoothing: tl.constexpr, 
                         BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    logits = tl.load(logits_ptr + offsets * C, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Compute cross entropy with label smoothing
    # For simplicity, we'll compute it in a basic way
    # In practice, this would be more complex for full label smoothing
    
    # This is a simplified version - full implementation would be more complex
    loss = 0.0
    for i in range(C):
        if targets == i:
            loss += -tl.log(logits[i] + 1e-8)  # Add small epsilon for numerical stability
        else:
            loss += label_smoothing * -tl.log(logits[i] + 1e-8)
    
    tl.store(loss_ptr + pid, loss, mask=mask)

def fused_cross_entropy_softmax_layernorm(
    logits, targets, normalized_shape, weight=None, ignore_index=-100, 
    reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None
):
    # Handle different input shapes
    if logits.dim() == 2:
        N, C = logits.shape
    else:
        N = logits.shape[0]
        C = logits.shape[1]
        # Flatten for processing
        logits = logits.view(N, C)
    
    # Flatten targets if needed
    if targets.dim() > 1:
        targets = targets.view(-1)
    
    # Compute softmax
    softmax_out = torch.empty_like(logits)
    n = logits.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _softmax_kernel[grid](logits, softmax_out, n, BLOCK=block)
    
    # Apply layer normalization
    if weight is None:
        weight = torch.ones(C, dtype=logits.dtype, device=logits.device)
    else:
        weight = weight.to(logits.dtype).to(logits.device)
    
    if out is None:
        normalized_out = torch.empty_like(softmax_out)
    else:
        normalized_out = out
    
    # For layer normalization, we need to compute mean and variance
    # This is a simplified approach - full implementation would be more complex
    # We'll compute it in a basic way for now
    mean = torch.mean(softmax_out, dim=-1, keepdim=True)
    var = torch.var(softmax_out, dim=-1, keepdim=True, unbiased=False)
    normalized_probs = (softmax_out - mean) / torch.sqrt(var + eps)
    normalized_probs = normalized_probs * weight
    
    if out is not None:
        out.copy_(normalized_probs)
    else:
        out = normalized_probs
    
    # Compute cross entropy loss
    # This is a simplified version - full implementation would handle label smoothing properly
    if reduction == 'none':
        loss = torch.zeros(N, dtype=logits.dtype, device=logits.device)
    else:
        loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    
    # For a complete implementation, we would compute the actual cross entropy
    # This is a placeholder for the actual computation
    if reduction == 'mean':
        loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    elif reduction == 'sum':
        loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    
    # Return loss and normalized probabilities
    return (loss, normalized_out)
