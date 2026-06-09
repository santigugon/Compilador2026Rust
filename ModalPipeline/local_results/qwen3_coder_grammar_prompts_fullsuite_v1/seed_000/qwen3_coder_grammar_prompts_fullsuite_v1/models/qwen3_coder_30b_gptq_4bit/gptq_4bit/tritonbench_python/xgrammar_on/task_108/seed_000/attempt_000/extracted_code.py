import torch
import triton
import triton.language as tl

def grid_sample_with_affine(input: torch.Tensor, theta: torch.Tensor, size: torch.Size, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor:
    # Validate inputs
    assert input.dim() == 4, "input must be 4D tensor"
    assert theta.dim() == 3, "theta must be 3D tensor"
    assert theta.size(1) == 2 and theta.size(2) == 3, "theta must be of shape (N, 2, 3)"
    assert input.size(0) == theta.size(0), "Batch size of input and theta must match"
    
    N, C, H_in, W_in = input.shape
    H_out, W_out = size[2], size[3]
    
    # Create output tensor
    output = torch.empty(N, C, H_out, W_out, device=input.device, dtype=input.dtype)
    
    # Generate grid using affine transformation
    grid = torch.nn.functional.affine_grid(theta, (N, 1, H_out, W_out), align_corners=align_corners)
    
    # Apply grid sampling
    if mode == 'bilinear':
        output = torch.nn.functional.grid_sample(input, grid, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners)
    elif mode == 'nearest':
        output = torch.nn.functional.grid_sample(input, grid, mode='nearest', padding_mode=padding_mode, align_corners=align_corners)
    elif mode == 'bicubic':
        output = torch.nn.functional.grid_sample(input, grid, mode='bicubic', padding_mode=padding_mode, align_corners=align_corners)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return output