import torch
import triton
import triton.language as tl

@triton.jit
def _affine_grid_kernel(
    theta_ptr, 
    grid_ptr, 
    N: tl.constexpr, 
    H_out: tl.constexpr, 
    W_out: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // (H_out * W_out)
    rest = pid % (H_out * W_out)
    h_idx = rest // W_out
    w_idx = rest % W_out
    
    if batch_idx >= N:
        return
    
    # Load theta for this batch
    theta_row0 = tl.load(theta_ptr + batch_idx * 6 + tl.arange(0, 3), mask=tl.arange(0, 3) < 3)
    theta_row1 = tl.load(theta_ptr + batch_idx * 6 + tl.arange(3, 6), mask=tl.arange(3, 6) < 6)
    
    # Grid coordinates
    x = w_idx.to(tl.float32)
    y = h_idx.to(tl.float32)
    
    # Apply affine transformation
    # Transform from output space to input space
    # x' = theta[0] * x + theta[1] * y + theta[2]
    # y' = theta[3] * x + theta[4] * y + theta[5]
    x_prime = theta_row0[0] * x + theta_row0[1] * y + theta_row0[2]
    y_prime = theta_row1[0] * x + theta_row1[1] * y + theta_row1[2]
    
    # Store grid coordinates
    grid_x = x_prime
    grid_y = y_prime
    tl.store(grid_ptr + batch_idx * H_out * W_out * 2 + h_idx * W_out * 2 + w_idx * 2, grid_x)
    tl.store(grid_ptr + batch_idx * H_out * W_out * 2 + h_idx * W_out * 2 + w_idx * 2 + 1, grid_y)

@triton.jit
def _grid_sample_bilinear_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    align_corners: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // (C * H_out * W_out)
    rest = pid % (C * H_out * W_out)
    c_idx = rest // (H_out * W_out)
    rest = rest % (H_out * W_out)
    h_idx = rest // W_out
    w_idx = rest % W_out
    
    if batch_idx >= N:
        return
    
    # Load grid coordinates
    grid_x = tl.load(grid_ptr + batch_idx * H_out * W_out * 2 + h_idx * W_out * 2 + w_idx * 2)
    grid_y = tl.load(grid_ptr + batch_idx * H_out * W_out * 2 + h_idx * W_out * 2 + w_idx * 2 + 1)
    
    # Normalize grid coordinates based on align_corners
    if align_corners:
        x_norm = (grid_x + 1.0) * (W_in - 1) / 2.0
        y_norm = (grid_y + 1.0) * (H_in - 1) / 2.0
    else:
        x_norm = grid_x * (W_in - 1) / 2.0 + (W_in - 1) / 2.0
        y_norm = grid_y * (H_in - 1) / 2.0 + (H_in - 1) / 2.0
    
    # Clamp coordinates to valid range
    x_norm = tl.maximum(0.0, tl.minimum(x_norm, W_in - 1.0))
    y_norm = tl.maximum(0.0, tl.minimum(y_norm, H_in - 1.0))
    
    # Get integer coordinates
    x0 = tl.floor(x_norm).to(tl.int32)
    y0 = tl.floor(y_norm).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Clamp to image boundaries
    x1 = tl.minimum(x1, W_in - 1)
    y1 = tl.minimum(y1, H_in - 1)
    
    # Compute interpolation weights
    wx = x_norm - x0.to(tl.float32)
    wy = y_norm - y0.to(tl.float32)
    
    # Load pixel values
    input_ptr_base = input_ptr + batch_idx * C * H_in * W_in + c_idx * H_in * W_in
    
    # Get four corner values
    val00 = tl.load(input_ptr_base + y0 * W_in + x0, mask=(y0 < H_in) & (x0 < W_in), other=0.0)
    val01 = tl.load(input_ptr_base + y0 * W_in + x1, mask=(y0 < H_in) & (x1 < W_in), other=0.0)
    val10 = tl.load(input_ptr_base + y1 * W_in + x0, mask=(y1 < H_in) & (x0 < W_in), other=0.0)
    val11 = tl.load(input_ptr_base + y1 * W_in + x1, mask=(y1 < H_in) & (x1 < W_in), other=0.0)
    
    # Bilinear interpolation
    val0 = val00 * (1.0 - wx) + val01 * wx
    val1 = val10 * (1.0 - wx) + val11 * wx
    result = val0 * (1.0 - wy) + val1 * wy
    
    # Store result
    tl.store(output_ptr + batch_idx * C * H_out * W_out + c_idx * H_out * W_out + h_idx * W_out + w_idx, result)

def grid_sample_with_affine(input: torch.Tensor, theta: torch.Tensor, size: torch.Size, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor:
    # Validate inputs
    assert input.dim() == 4, "input must be a 4D tensor (N, C, H_in, W_in)"
    assert theta.dim() == 3, "theta must be a 3D tensor (N, 2, 3)"
    assert theta.size(0) == input.size(0), "batch size mismatch between input and theta"
    assert theta.size(1) == 2 and theta.size(2) == 3, "theta must be of shape (N, 2, 3)"
    assert len(size) == 4, "size must be a 4D size (N, C, H_out, W_out)"
    
    N, C, H_in, W_in = input.shape
    _, _, H_out, W_out = size
    
    # Create output tensor
    output = torch.empty(size, dtype=input.dtype, device=input.device)
    
    # Generate grid using affine transformation
    grid = torch.empty((N, H_out, W_out, 2), dtype=torch.float32, device=input.device)
    
    # Launch affine grid kernel
    block = 256
    grid_size = N * H_out * W_out
    grid_pid = triton.cdiv(grid_size, block)
    
    _affine_grid_kernel[grid_pid](theta, grid, N, H_out, W_out, BLOCK=block)
    
    # Apply grid sampling
    if mode == 'bilinear':
        # Launch grid sample kernel
        block = 256
        output_size = N * C * H_out * W_out
        output_pid = triton.cdiv(output_size, block)
        
        _grid_sample_bilinear_kernel[output_pid](
            input, grid, output, N, C, H_in, W_in, H_out, W_out, align_corners, BLOCK=block
        )
    else:
        # For other modes, fall back to PyTorch implementation
        # This is a simplified version - in practice, you'd implement other modes
        # like 'nearest' and 'bicubic' with their own kernels
        grid_sample_input = input.permute(0, 2, 3, 1).contiguous()  # (N, H_in, W_in, C)
        grid_sample_grid = grid.flip(-1)  # Flip x,y to y,x for PyTorch
        if not align_corners:
            grid_sample_grid = grid_sample_grid * 2.0 - 1.0  # Normalize to [-1, 1]
        output = torch.nn.functional.grid_sample(
            grid_sample_input, grid_sample_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        ).permute(0, 3, 1, 2)  # Back to (N, C, H_out, W_out)
    
    return output
