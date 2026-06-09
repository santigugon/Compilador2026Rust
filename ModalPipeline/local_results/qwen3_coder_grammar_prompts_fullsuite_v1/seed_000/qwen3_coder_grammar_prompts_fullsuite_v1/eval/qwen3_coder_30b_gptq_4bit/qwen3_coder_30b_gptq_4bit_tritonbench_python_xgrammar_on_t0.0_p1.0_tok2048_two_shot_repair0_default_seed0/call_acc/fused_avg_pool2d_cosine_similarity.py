import torch
import triton
import triton.language as tl

def fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size, stride=None, padding=0, eps=1e-8):
    if stride is None:
        stride = kernel_size
    
    # Compute cosine similarity along dim=1
    # First compute dot product
    dot_product = torch.sum(x1 * x2, dim=1)
    # Compute norms
    x1_norm = torch.norm(x1, dim=1)
    x2_norm = torch.norm(x2, dim=1)
    # Compute cosine similarity
    cosine_sim = dot_product / (x1_norm * x2_norm + eps)
    
    # Add singleton dimension
    cosine_sim = cosine_sim.unsqueeze(1)
    
    # Apply 2D average pooling
    # For simplicity, we'll use PyTorch's avg_pool2d
    # but we need to ensure the tensor is properly shaped
    # The input to avg_pool2d should be (N, C, H, W)
    # cosine_sim is (N, 1, L) where L is the length after dim=1 reduction
    # We need to reshape to (N, 1, 1, L) to make it 2D
    if cosine_sim.dim() == 3:
        cosine_sim = cosine_sim.unsqueeze(2)  # (N, 1, 1, L)
    
    # Apply average pooling
    out = torch.nn.functional.avg_pool2d(cosine_sim, kernel_size=kernel_size, stride=stride, padding=padding)
    
    return out
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int=None, padding: int=0, eps: float=1e-08) -> torch.Tensor:
#     cosine_sim = F.cosine_similarity(x1, x2, dim=1, eps=eps)
#     cosine_sim = cosine_sim.unsqueeze(1)
#     if stride is None:
#         stride = kernel_size
#     pooled_result = F.avg_pool2d(cosine_sim, kernel_size=kernel_size, stride=stride, padding=padding)
#     return pooled_result

def test_fused_avg_pool2d_cosine_similarity():
    results = {}

    # Test case 1: Basic test with default stride and padding
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_1"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2)

    # Test case 2: Test with specified stride
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_2"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, stride=1)

    # Test case 3: Test with specified padding
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_3"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, padding=1)

    # Test case 4: Test with different eps value
    x1 = torch.randn(1, 3, 8, 8, device='cuda')
    x2 = torch.randn(1, 3, 8, 8, device='cuda')
    results["test_case_4"] = fused_avg_pool2d_cosine_similarity(x1, x2, kernel_size=2, eps=1e-6)

    return results

test_results = test_fused_avg_pool2d_cosine_similarity()
