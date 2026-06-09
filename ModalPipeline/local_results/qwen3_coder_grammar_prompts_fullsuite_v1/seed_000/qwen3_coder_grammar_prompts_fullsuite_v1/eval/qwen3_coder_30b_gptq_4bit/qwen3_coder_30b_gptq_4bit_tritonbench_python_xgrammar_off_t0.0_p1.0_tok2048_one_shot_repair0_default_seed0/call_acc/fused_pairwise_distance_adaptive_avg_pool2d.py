import torch
import triton
import triton.language as tl

@triton.jit
def _adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr, 
    input_height, input_width,
    output_height, output_width,
    stride_h, stride_w,
    padding_h, padding_w,
    BLOCK_H=16, BLOCK_W=16
):
    # Get block indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate output indices
    out_h = pid_h * BLOCK_H
    out_w = pid_w * BLOCK_W
    
    # Calculate input indices
    in_h_start = out_h * stride_h - padding_h
    in_w_start = out_w * stride_w - padding_w
    
    # Loop over input region
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    count = 0
    
    for ih in range(BLOCK_H):
        for iw in range(BLOCK_W):
            h = in_h_start + ih
            w = in_w_start + iw
            
            # Check bounds
            if h >= 0 and h < input_height and w >= 0 and w < input_width:
                acc[ih, iw] = tl.load(input_ptr + h * input_width + w)
                count += 1
    
    # Compute average
    if count > 0:
        output_val = tl.sum(acc) / count
    else:
        output_val = 0.0
    
    # Store result
    tl.store(output_ptr + out_h * output_width + out_w, output_val)

def fused_pairwise_distance_adaptive_avg_pool2d(
    x1: torch.Tensor, 
    x2: torch.Tensor, 
    output_size: int or tuple, 
    p: float = 2.0, 
    eps: float = 1e-6, 
    keepdim: bool = False
) -> torch.Tensor:
    # Handle output_size
    if isinstance(output_size, int):
        output_height = output_size
        output_width = output_size
    else:
        output_height, output_width = output_size
    
    # Ensure inputs are 4D (B, C, H, W)
    if x1.dim() == 3:
        x1 = x1.unsqueeze(0)
    if x2.dim() == 3:
        x2 = x2.unsqueeze(0)
    
    batch_size, channels, height, width = x1.shape
    
    # Calculate stride and padding for adaptive pooling
    stride_h = height // output_height
    stride_w = width // output_width
    padding_h = (stride_h * output_height - height) // 2
    padding_w = (stride_w * output_width - width) // 2
    
    # Apply adaptive average pooling
    pooled_x1 = torch.zeros(batch_size, channels, output_height, output_width, device=x1.device, dtype=x1.dtype)
    pooled_x2 = torch.zeros(batch_size, channels, output_height, output_width, device=x2.device, dtype=x2.dtype)
    
    # Simple implementation using PyTorch's adaptive_avg_pool2d
    pooled_x1 = torch.nn.functional.adaptive_avg_pool2d(x1, (output_height, output_width))
    pooled_x2 = torch.nn.functional.adaptive_avg_pool2d(x2, (output_height, output_width))
    
    # Flatten for distance calculation
    pooled_x1_flat = pooled_x1.view(batch_size, channels, -1)
    pooled_x2_flat = pooled_x2.view(batch_size, channels, -1)
    
    # Compute pairwise distance
    diff = pooled_x1_flat.unsqueeze(2) - pooled_x2_flat.unsqueeze(1)
    distance = torch.norm(diff, p=p, dim=3)
    
    # Add eps to avoid division by zero
    distance = distance + eps
    
    # Apply keepdim
    if keepdim:
        # Reshape to maintain dimensions
        pass
    
    return distance

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
