import torch
import triton
import triton.language as tl

@triton.jit
def cosine_embedding_loss_kernel(
    input1_ptr, input2_ptr, target_ptr, output_ptr,
    n_features, n_samples,
    margin,
    reduction,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    indices = offset + tl.arange(0, BLOCK_SIZE)
    
    # Load data
    input1 = tl.load(input1_ptr + indices, mask=indices < n_features)
    input2 = tl.load(input2_ptr + indices, mask=indices < n_features)
    target = tl.load(target_ptr + pid, mask=pid < n_samples)
    
    # Compute dot product
    dot_product = tl.sum(input1 * input2)
    
    # Compute L2 norms
    norm1 = tl.sqrt(tl.sum(input1 * input1) + 1e-8)
    norm2 = tl.sqrt(tl.sum(input2 * input2) + 1e-8)
    
    # Compute cosine similarity
    cos_sim = dot_product / (norm1 * norm2)
    
    # Compute loss
    loss = tl.where(target == 1, 1 - cos_sim, tl.maximum(0, cos_sim - margin))
    
    # Store result
    tl.store(output_ptr + pid, loss)

def fused_cosine_embedding_loss_with_normalization(
    input1: torch.Tensor, 
    input2: torch.Tensor, 
    target: torch.Tensor, 
    margin: float = 0, 
    reduction: str = 'mean'
) -> torch.Tensor:
    # Validate inputs
    assert input1.shape == input2.shape, "input1 and input2 must have the same shape"
    assert target.shape[0] == input1.shape[0], "target must have the same batch size as input tensors"
    assert target.dtype == torch.long or target.dtype == torch.int, "target must be of integer type"
    
    # Flatten input tensors for processing
    input1_flat = input1.view(input1.shape[0], -1)
    input2_flat = input2.view(input2.shape[0], -1)
    
    n_samples, n_features = input1_flat.shape
    
    # Normalize inputs
    input1_norm = input1_flat / (torch.norm(input1_flat, p=2, dim=1, keepdim=True) + 1e-8)
    input2_norm = input2_flat / (torch.norm(input2_flat, p=2, dim=1, keepdim=True) + 1e-8)
    
    # Prepare output tensor
    output = torch.empty(n_samples, dtype=torch.float32, device=input1.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_samples, BLOCK_SIZE),)
    
    cosine_embedding_loss_kernel[grid](
        input1_norm, input2_norm, target, output,
        n_features, n_samples,
        margin,
        reduction,
        BLOCK_SIZE
    )
    
    # Apply reduction
    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:  # reduction == 'none'
        return output

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
