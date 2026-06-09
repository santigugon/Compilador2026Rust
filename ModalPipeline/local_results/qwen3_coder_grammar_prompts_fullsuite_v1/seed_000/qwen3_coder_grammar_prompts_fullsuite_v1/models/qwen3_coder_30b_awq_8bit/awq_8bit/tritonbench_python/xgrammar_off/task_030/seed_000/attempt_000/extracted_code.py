import torch
import triton
import triton.language as tl

@triton.jit
def _fused_cross_entropy_softmax_layernorm_kernel(
    logits_ptr, targets_ptr, weight_ptr, out_ptr,
    N, C, reduction, ignore_index, label_smoothing, eps,
    logits_stride_0, logits_stride_1,
    targets_stride_0,
    weight_stride_0,
    out_stride_0, out_stride_1,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    batch_offset = pid * BLOCK
    
    # Load logits for this batch
    offsets = batch_offset + tl.arange(0, BLOCK)
    mask = offsets < N
    
    # Load targets
    targets = tl.load(targets_ptr + offsets, mask=mask, other=ignore_index)
    
    # Initialize loss accumulator
    loss = tl.zeros((BLOCK,), dtype=tl.float32)
    
    # Compute cross-entropy loss
    for i in range(C):
        # Load logits
        logits = tl.load(logits_ptr + offsets * logits_stride_0 + i * logits_stride_1, mask=mask, other=0.0)
        
        # Apply softmax
        max_logits = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - max_logits)
        sum_exp_logits = tl.sum(exp_logits, axis=0)
        softmax_probs = exp_logits / (sum_exp_logits + eps)
        
        # Apply weight if provided
        if weight_ptr is not None:
            weight = tl.load(weight_ptr + i * weight_stride_0, mask=(i < C), other=1.0)
            softmax_probs = softmax_probs * weight
        
        # Compute loss
        if label_smoothing > 0:
            # Apply label smoothing
            smooth_loss = -tl.log(softmax_probs + eps) * (1 - label_smoothing) / C
            # Add smoothing term
            smooth_loss += -tl.log(1.0 / C) * label_smoothing
            # Apply to targets
            target_mask = targets == i
            loss += tl.where(target_mask, smooth_loss, 0.0)
        else:
            # Standard cross-entropy
            target_mask = targets == i
            loss += tl.where(target_mask, -tl.log(softmax_probs + eps), 0.0)
    
    # Apply reduction
    if reduction == 0:  # 'none'
        pass  # No reduction
    elif reduction == 1:  # 'mean'
        loss = loss / N
    elif reduction == 2:  # 'sum'
        loss = tl.sum(loss, axis=0)
    
    # Store loss
    tl.store(out_ptr + offsets, loss, mask=mask)

def fused_cross_entropy_softmax_layernorm(
    logits, targets, normalized_shape, weight=None, ignore_index=-100, 
    reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None
):
    # Handle different input shapes
    if logits.dim() == 2:
        N, C = logits.shape
    else:
        # For higher dimensional inputs, flatten to (N, C)
        N = logits.shape[0]
        C = logits.shape[1]
        logits = logits.view(N, C)
    
    # Handle targets shape
    if targets.dim() == 1:
        targets = targets.view(N)
    else:
        targets = targets.view(N)
    
    # Handle weight
    if weight is not None:
        if weight.shape[0] != C:
            raise ValueError(f"weight must have size {C} in dimension 0")
    else:
        weight = torch.ones(C, dtype=logits.dtype, device=logits.device)
    
    # Handle reduction
    reduction_map = {'none': 0, 'mean': 1, 'sum': 2}
    if reduction not in reduction_map:
        raise ValueError(f"reduction must be 'none', 'mean', or 'sum'")
    reduction_code = reduction_map[reduction]
    
    # Prepare output tensor
    if out is not None:
        out_probs = out
    else:
        out_probs = torch.empty_like(logits)
    
    # Compute softmax probabilities
    # First, compute max for numerical stability
    max_logits = logits.max(dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits - max_logits)
    sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)
    softmax_probs = exp_logits / sum_exp_logits
    
    # Apply layer normalization
    mean = softmax_probs.mean(dim=1, keepdim=True)
    var = softmax_probs.var(dim=1, keepdim=True, unbiased=False)
    normalized_probs = (softmax_probs - mean) / torch.sqrt(var + eps)
    
    # Compute cross-entropy loss
    if label_smoothing > 0:
        # Apply label smoothing
        targets_one_hot = torch.zeros_like(softmax_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        smooth_targets = targets_one_hot * (1 - label_smoothing) + label_smoothing / C
        loss = -torch.sum(smooth_targets * torch.log(softmax_probs + eps), dim=1)
    else:
        # Standard cross-entropy
        loss = -torch.log(softmax_probs.gather(1, targets.unsqueeze(1)).squeeze(1) + eps)
    
    # Apply reduction
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    # Return loss and normalized probabilities
    return loss, normalized_probs
