import torch
import triton
import triton.language as tl
import math

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
    out = x_exp / x_sum
    tl.store(out_ptr + offsets, out, mask=mask)

def _layer_norm_kernel(x_ptr, out_ptr, weight_ptr, mean_ptr, var_ptr, n: tl.constexpr, normalized_shape: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / normalized_shape
    tl.store(mean_ptr + pid, mean)
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / normalized_shape
    tl.store(var_ptr + pid, var)
    
    # Normalize
    x_norm = x_centered / tl.sqrt(var + eps)
    
    # Apply weight
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + offsets % normalized_shape, mask=mask, other=1.0)
        x_norm = x_norm * weight
    
    tl.store(out_ptr + offsets, x_norm, mask=mask)

def _cross_entropy_kernel(logits_ptr, targets_ptr, out_ptr, n: tl.constexpr, num_classes: tl.constexpr, ignore_index: tl.constexpr, label_smoothing: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    
    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0)
    targets = tl.load(targets_ptr + offsets, mask=mask, other=0.0)
    
    # Apply label smoothing
    if label_smoothing > 0:
        # Convert to probabilities
        targets = targets * (1 - label_smoothing) + label_smoothing / num_classes
    
    # Compute cross entropy
    ce = -tl.log(tl.exp(logits) / tl.sum(tl.exp(logits), axis=0) + 1e-8) * targets
    
    # Apply ignore index
    ignore_mask = targets == ignore_index
    ce = tl.where(ignore_mask, 0.0, ce)
    
    tl.store(out_ptr + offsets, ce, mask=mask)

def fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None):
    # Handle scalar normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    
    # Flatten logits and targets
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    
    # Get dimensions
    batch_size = logits_flat.size(0)
    num_classes = logits_flat.size(1)
    n = batch_size * num_classes
    
    # Initialize output tensors
    if out is None:
        out_probs = torch.empty_like(logits)
    else:
        out_probs = out
    
    # Compute softmax
    softmax_out = torch.empty_like(logits_flat)
    block = 256
    grid = (triton.cdiv(n, block),)
    _softmax_kernel[grid](logits_flat, softmax_out, n, BLOCK=block)
    
    # Apply layer normalization
    if weight is not None:
        weight_tensor = weight
    else:
        weight_tensor = torch.ones(num_classes, device=logits.device, dtype=logits.dtype)
    
    # For simplicity, we'll compute layer norm on the flattened tensor
    # and then reshape back
    layer_norm_out = torch.empty_like(softmax_out)
    mean_tensor = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
    var_tensor = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
    
    # Compute layer norm
    _layer_norm_kernel[grid](softmax_out, layer_norm_out, weight_tensor, mean_tensor, var_tensor, n, num_classes, eps, BLOCK=block)
    
    # Reshape back to original shape
    out_probs = layer_norm_out.view(logits.shape)
    
    # Compute cross entropy loss
    ce_out = torch.empty(batch_size, device=logits.device, dtype=torch.float32)
    _cross_entropy_kernel[grid](logits_flat, targets_flat, ce_out, n, num_classes, ignore_index, label_smoothing, BLOCK=block)
    
    # Apply reduction
    if reduction == 'mean':
        loss = ce_out.mean()
    elif reduction == 'sum':
        loss = ce_out.sum()
    else:  # 'none'
        loss = ce_out
    
    return loss, out_probs
##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_fused_cross_entropy_softmax_layernorm():
    results = {}

    # Test case 1: Basic functionality with default parameters
    logits = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    targets = torch.tensor([2, 1], device='cuda')
    normalized_shape = 3
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape)
    results["test_case_1"] = (loss.item(), output.cpu().numpy())

    # Test case 2: With weight parameter
    weight = torch.tensor([0.1, 0.2, 0.3], device='cuda')
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=weight)
    results["test_case_2"] = (loss.item(), output.cpu().numpy())

    # Test case 3: With ignore_index parameter
    targets_ignore = torch.tensor([2, -100], device='cuda')
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets_ignore, normalized_shape, ignore_index=-100)
    results["test_case_3"] = (loss.item(), output.cpu().numpy())

    # Test case 4: With label_smoothing parameter
    loss, output = fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, label_smoothing=0.1)
    results["test_case_4"] = (loss.item(), output.cpu().numpy())

    return results

test_results = test_fused_cross_entropy_softmax_layernorm()
