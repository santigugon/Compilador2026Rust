import torch
import triton
import triton.language as tl

@triton.jit
def fused_cross_entropy_softmax_layernorm_kernel(
    logits_ptr, targets_ptr, weight_ptr, out_ptr,
    N, C, reduction, label_smoothing, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load logits and targets
    logits = tl.load(logits_ptr + offsets[:, None] * C + tl.arange(0, C)[None, :], mask=mask[:, None])
    targets = tl.load(targets_ptr + offsets, mask=mask)
    
    # Softmax computation
    max_logits = tl.max(logits, axis=1, keepdims=True)
    shifted_logits = logits - max_logits
    exp_logits = tl.exp(shifted_logits)
    sum_exp_logits = tl.sum(exp_logits, axis=1, keepdims=True)
    softmax_probs = exp_logits / (sum_exp_logits + eps)
    
    # Cross-entropy loss computation
    if label_smoothing > 0:
        # Apply label smoothing
        num_classes = C
        smooth_loss = -tl.log(softmax_probs + eps)
        ce_loss = tl.sum(smooth_loss * targets, axis=1)
        # Apply label smoothing
        ce_loss = (1 - label_smoothing) * ce_loss + label_smoothing * tl.sum(softmax_probs, axis=1) / num_classes
    else:
        # Standard cross-entropy
        ce_loss = -tl.log(softmax_probs[tl.arange(0, N), targets] + eps)
    
    # Layer normalization
    mean = tl.mean(softmax_probs, axis=1, keepdims=True)
    var = tl.var(softmax_probs, axis=1, keepdims=True)
    normalized = (softmax_probs - mean) / tl.sqrt(var + eps)
    
    # Reduction
    if reduction == "mean":
        ce_loss = tl.sum(ce_loss) / N
    elif reduction == "sum":
        ce_loss = tl.sum(ce_loss)
    
    # Store results
    tl.store(out_ptr + offsets, normalized, mask=mask)

def fused_cross_entropy_softmax_layernorm(
    logits, targets, normalized_shape, weight=None, ignore_index=-100, 
    reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None
):
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    
    # Flatten logits and targets
    if logits.dim() > 2:
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
    
    N, C = logits.shape
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(logits, dtype=torch.float32)
    
    # Prepare weight tensor if needed
    if weight is not None:
        weight = weight.to(logits.dtype)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_cross_entropy_softmax_layernorm_kernel[
        num_blocks
    ](
        logits, targets, weight, out,
        N, C, reduction, label_smoothing, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return loss and normalized probabilities
    loss = torch.tensor(0.0, dtype=torch.float32, device=logits.device)
    if reduction == "mean":
        loss = torch.sum(-torch.log(torch.gather(torch.softmax(logits, dim=1), 1, targets.unsqueeze(1)).squeeze(1) + eps)) / N
    elif reduction == "sum":
        loss = torch.sum(-torch.log(torch.gather(torch.softmax(logits, dim=1), 1, targets.unsqueeze(1)).squeeze(1) + eps))
    
    return loss, out
