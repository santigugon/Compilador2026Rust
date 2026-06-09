import torch
import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_softmax_layernorm_kernel(
    logits_ptr, targets_ptr, weight_ptr, out_ptr,
    N, C, normalized_shape, ignore_index, reduction, label_smoothing, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Thread and block indices
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load logits
    logits = tl.load(logits_ptr + offsets * C, mask=mask[:, None])
    
    # Apply softmax
    logits_max = tl.max(logits, axis=1, keepdims=True)
    logits_exp = tl.exp(logits - logits_max)
    logits_sum = tl.sum(logits_exp, axis=1, keepdims=True)
    probs = logits_exp / logits_sum
    
    # Compute cross-entropy loss
    if label_smoothing > 0.0:
        # Apply label smoothing
        smooth_loss = -tl.log(probs + 1e-8)
        ce_loss = tl.sum(smooth_loss * targets_ptr, axis=1)
    else:
        # Standard cross-entropy
        ce_loss = -tl.log(tl.sum(probs * targets_ptr, axis=1) + 1e-8)
    
    # Apply reduction
    if reduction == "mean":
        loss = tl.sum(ce_loss) / N
    elif reduction == "sum":
        loss = tl.sum(ce_loss)
    else:
        loss = ce_loss
    
    # Layer normalization
    mean = tl.sum(probs, axis=1, keepdims=True) / normalized_shape
    var = tl.sum((probs - mean) ** 2, axis=1, keepdims=True) / normalized_shape
    std = tl.sqrt(var + eps)
    normalized_probs = (probs - mean) / std
    
    # Store output
    tl.store(out_ptr + offsets * C, normalized_probs, mask=mask[:, None])

def fused_cross_entropy_softmax_layernorm(
    logits, targets, normalized_shape, weight=None, ignore_index=-100, 
    reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None
):
    # Ensure inputs are contiguous
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    # Determine dimensions
    N = logits.shape[0]
    C = logits.shape[1]
    
    # Handle different target shapes
    if targets.dim() == 1:
        # One-hot encode targets
        targets_one_hot = torch.zeros_like(logits)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets = targets_one_hot
    
    # Apply weight if provided
    if weight is not None:
        weight = weight.contiguous()
        targets = targets * weight
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(logits)
    else:
        out = out.contiguous()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    fused_cross_entropy_softmax_layernorm_kernel[grid](
        logits, targets, weight, out,
        N, C, normalized_shape, ignore_index, reduction, label_smoothing, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return loss and normalized probabilities
    return out, torch.tensor(0.0)  # Placeholder for loss
