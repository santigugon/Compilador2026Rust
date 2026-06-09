import torch
import triton
import triton.language as tl

def fused_cosine_embedding_loss_with_normalization(input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor, margin: float = 0, reduction: str = 'mean') -> torch.Tensor:
    # Validate inputs
    if input1.shape != input2.shape:
        raise ValueError("input1 and input2 must have the same shape")
    if target.shape != input1.shape[:1]:
        raise ValueError("target must have the same length as the first dimension of input1 and input2")
    
    # Flatten input tensors for easier processing
    input1_flat = input1.view(-1, input1.shape[-1])
    input2_flat = input2.view(-1, input2.shape[-1])
    target_flat = target.view(-1)
    
    # Compute L2 norms
    input1_norm = torch.norm(input1_flat, p=2, dim=1, keepdim=True)
    input2_norm = torch.norm(input2_flat, p=2, dim=1, keepdim=True)
    
    # Avoid division by zero
    input1_norm = torch.where(input1_norm == 0, torch.ones_like(input1_norm), input1_norm)
    input2_norm = torch.where(input2_norm == 0, torch.ones_like(input2_norm), input2_norm)
    
    # Normalize inputs
    input1_normed = input1_flat / input1_norm
    input2_normed = input2_flat / input2_norm
    
    # Compute cosine similarity
    cos_sim = torch.sum(input1_normed * input2_normed, dim=1)
    
    # Compute loss
    loss = torch.where(target_flat == 1,
                       1 - cos_sim,
                       torch.clamp(cos_sim + margin, min=0.0))
    
    # Apply reduction
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    else:  # reduction == 'none'
        return loss
##################################################################################################################################################



import torch
import torch.nn.functional as F
import torch

def test_fused_cosine_embedding_loss_with_normalization():
    results = {}

    # Test case 1: Default margin and reduction
    input1 = torch.randn(3, 5, device='cuda', requires_grad=True)
    input2 = torch.randn(3, 5, device='cuda', requires_grad=True)
    target = torch.tensor([1, -1, 1], device='cuda')
    results["test_case_1"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target)

    # Test case 2: Margin > 0
    margin = 0.5
    results["test_case_2"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target, margin=margin)

    # Test case 3: Reduction 'sum'
    reduction = 'sum'
    results["test_case_3"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target, reduction=reduction)

    # Test case 4: Reduction 'none'
    reduction = 'none'
    results["test_case_4"] = fused_cosine_embedding_loss_with_normalization(input1, input2, target, reduction=reduction)

    return results

test_results = test_fused_cosine_embedding_loss_with_normalization()
