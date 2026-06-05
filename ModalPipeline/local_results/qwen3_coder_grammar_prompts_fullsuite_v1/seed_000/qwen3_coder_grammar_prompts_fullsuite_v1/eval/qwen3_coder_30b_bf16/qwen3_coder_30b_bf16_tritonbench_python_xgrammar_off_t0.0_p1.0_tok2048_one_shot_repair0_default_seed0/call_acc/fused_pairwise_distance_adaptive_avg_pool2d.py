import torch
import triton
import triton.language as tl

@triton.jit
def _pairwise_distance_kernel(
    x1_ptr, x2_ptr, out_ptr,
    x1_stride_0, x1_stride_1, x1_stride_2, x1_stride_3,
    x2_stride_0, x2_stride_1, x2_stride_2, x2_stride_3,
    out_stride_0, out_stride_1,
    n1, n2, c, h, w,
    p, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // n2
    sample_idx = pid % n2
    
    if batch_idx >= n1 or sample_idx >= n2:
        return
    
    # Compute the distance between x1[batch_idx] and x2[sample_idx]
    acc = 0.0
    for i in range(c * h * w):
        x1_val = tl.load(x1_ptr + batch_idx * x1_stride_0 + i * x1_stride_2)
        x2_val = tl.load(x2_ptr + sample_idx * x2_stride_0 + i * x2_stride_2)
        diff = x1_val - x2_val
        acc += tl.power(tl.abs(diff), p)
    
    dist = tl.power(acc + eps, 1.0 / p)
    
    tl.store(out_ptr + batch_idx * out_stride_0 + sample_idx * out_stride_1, dist)

def fused_pairwise_distance_adaptive_avg_pool2d(
    x1: torch.Tensor, 
    x2: torch.Tensor, 
    output_size: int or tuple, 
    p: float = 2.0, 
    eps: float = 1e-6, 
    keepdim: bool = False
) -> torch.Tensor:
    # Apply adaptive average pooling
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    
    # Ensure input tensors are on the same device and have correct dtype
    device = x1.device
    dtype = x1.dtype
    
    # Apply adaptive average pooling
    if x1.dim() == 4:
        x1_pooled = torch.nn.functional.adaptive_avg_pool2d(x1, output_size)
        x2_pooled = torch.nn.functional.adaptive_avg_pool2d(x2, output_size)
    else:
        raise ValueError("Input tensors must be 4D (N, C, H, W)")
    
    # Flatten the spatial dimensions for distance computation
    x1_flat = x1_pooled.view(x1_pooled.size(0), -1)
    x2_flat = x2_pooled.view(x2_pooled.size(0), -1)
    
    # Prepare output tensor
    n1, c1 = x1_flat.shape
    n2, c2 = x2_flat.shape
    
    if n1 != x1.size(0) or n2 != x2.size(0):
        raise ValueError("Mismatch in batch dimensions after pooling")
    
    # Compute pairwise distances using Triton kernel
    out = torch.empty(n1, n2, device=device, dtype=dtype)
    
    # Launch kernel
    grid = (n1 * n2,)
    BLOCK_SIZE = 1024
    
    _pairwise_distance_kernel[grid](
        x1_flat, x2_flat, out,
        x1_flat.stride(0), x1_flat.stride(1), 0, 0,
        x2_flat.stride(0), x2_flat.stride(1), 0, 0,
        out.stride(0), out.stride(1),
        n1, n2, c1, 1, 1,
        p, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if keepdim:
        out = out.unsqueeze(-1)
    
    return out

##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_pairwise_distance_adaptive_avg_pool2d(x1: torch.Tensor, x2: torch.Tensor, output_size: int or tuple, p: float=2.0, eps: float=1e-06, keepdim: bool=False) -> torch.Tensor:
#     pooled_x1 = F.adaptive_avg_pool2d(x1, output_size)
#     pooled_x2 = F.adaptive_avg_pool2d(x2, output_size)
#     diff = pooled_x1 - pooled_x2
#     dist = torch.norm(diff, p=p, dim=(1, 2, 3), keepdim=keepdim) + eps
#     return dist

def test_fused_pairwise_distance_adaptive_avg_pool2d():
    results = {}
    
    # Test case 1: Basic test with default parameters
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_1"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8))

    # Test case 2: Different output size
    x1 = torch.rand((2, 3, 64, 64), device='cuda')
    x2 = torch.rand((2, 3, 64, 64), device='cuda')
    results["test_case_2"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(16, 16))

    # Test case 3: Different norm degree
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_3"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8), p=1.0)

    # Test case 4: Keep dimension
    x1 = torch.rand((2, 3, 32, 32), device='cuda')
    x2 = torch.rand((2, 3, 32, 32), device='cuda')
    results["test_case_4"] = fused_pairwise_distance_adaptive_avg_pool2d(x1, x2, output_size=(8, 8), keepdim=True)

    return results

test_results = test_fused_pairwise_distance_adaptive_avg_pool2d()
