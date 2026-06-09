import torch
import triton
import triton.language as tl

def fused_cross_entropy_softmax_layernorm(logits, targets, normalized_shape, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0, eps=1e-5, *, out=None):
    # Handle scalar normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    
    # Flatten logits and targets for processing
    original_shape = logits.shape
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    
    # Flatten logits and targets
    logits_flat = logits.view(batch_size, -1)
    targets_flat = targets.view(batch_size, -1)
    
    # Determine output shape
    if out is None:
        out = torch.empty_like(logits)
    else:
        assert out.shape == logits.shape, "Output tensor must match logits shape"
    
    # Compute softmax and cross-entropy loss
    if reduction == 'none':
        # For 'none' reduction, we compute loss per sample
        loss = torch.empty(batch_size, dtype=torch.float32)
    else:
        loss = torch.empty((), dtype=torch.float32)
    
    # Apply softmax
    logits_softmax = torch.softmax(logits_flat, dim=1)
    
    # Apply layer normalization
    # Compute mean and variance for layer normalization
    mean = logits_softmax.mean(dim=1, keepdim=True)
    var = (logits_softmax - mean).pow(2).mean(dim=1, keepdim=True)
    std = (var + eps).sqrt()
    normalized_probs = (logits_softmax - mean) / std
    
    # Copy normalized probabilities to output
    out = normalized_probs.view(original_shape)
    
    # Compute cross-entropy loss
    if reduction == 'none':
        # For 'none' reduction, compute loss per sample
        loss = torch.nn.functional.cross_entropy(
            logits_flat, targets_flat.view(batch_size), 
            weight=weight, ignore_index=ignore_index, 
            reduction='none', label_smoothing=label_smoothing
        )
    else:
        # For 'mean' or 'sum' reduction
        loss = torch.nn.functional.cross_entropy(
            logits_flat, targets_flat.view(batch_size), 
            weight=weight, ignore_index=ignore_index, 
            reduction=reduction, label_smoothing=label_smoothing
        )
    
    return (loss, out)
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
