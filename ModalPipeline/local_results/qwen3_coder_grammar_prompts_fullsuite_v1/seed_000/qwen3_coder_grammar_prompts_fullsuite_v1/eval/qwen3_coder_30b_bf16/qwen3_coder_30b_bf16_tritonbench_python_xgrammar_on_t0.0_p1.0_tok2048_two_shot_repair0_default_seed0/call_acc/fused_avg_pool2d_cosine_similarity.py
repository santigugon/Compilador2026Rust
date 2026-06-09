import torch
import triton
import triton.language as tl

def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    if stride is None:
        stride = kernel_size
    
    # Compute cosine similarity along dim=1
    # First, compute dot product
    dot_product = torch.sum(x1 * x2, dim=1)
    
    # Compute norms
    norm_x1 = torch.norm(x1, p=2, dim=1)
    norm_x2 = torch.norm(x2, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = dot_product / (norm_x1 * norm_x2 + eps)
    
    # Add singleton dimension
    cosine_sim = cosine_sim.unsqueeze(1)
    
    # Apply 2D average pooling
    result = torch.nn.functional.avg_pool2d(
        cosine_sim, 
        kernel_size=(kernel_size, 1), 
        stride=(stride, 1), 
        padding=(padding, 0)
    )
    
    return result
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
