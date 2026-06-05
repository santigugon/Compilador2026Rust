import torch
import triton
import triton.language as tl

@triton.jit
def _affine_grid_kernel(theta_ptr, grid_ptr, N: tl.constexpr, H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // (H * W)
    y_idx = (pid % (H * W)) // W
    x_idx = (pid % (H * W)) % W
    
    if batch_idx >= N:
        return
    
    # Load theta for this batch
    theta_00 = tl.load(theta_ptr + batch_idx * 6 + 0)
    theta_01 = tl.load(theta_ptr + batch_idx * 6 + 1)
    theta_02 = tl.load(theta_ptr + batch_idx * 6 + 2)
    theta_10 = tl.load(theta_ptr + batch_idx * 6 + 3)
    theta_11 = tl.load(theta_ptr + batch_idx * 6 + 4)
    theta_12 = tl.load(theta_ptr + batch_idx * 6 + 5)
    
    # Normalize coordinates to [-1, 1]
    x = (x_idx - (W - 1) / 2.0) / ((W - 1) / 2.0)
    y = (y_idx - (H - 1) / 2.0) / ((H - 1) / 2.0)
    
    # Apply affine transformation
    grid_x = theta_00 * x + theta_01 * y + theta_02
    grid_y = theta_10 * x + theta_11 * y + theta_12
    
    # Store grid coordinates
    tl.store(grid_ptr + batch_idx * H * W * 2 + y_idx * W * 2 + x_idx * 2 + 0, grid_x)
    tl.store(grid_ptr + batch_idx * H * W * 2 + y_idx * W * 2 + x_idx * 2 + 1, grid_y)

@triton.jit
def _grid_sample_bilinear_kernel(input_ptr, grid_ptr, output_ptr, 
                                N: tl.constexpr, C: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
                                H_in: tl.constexpr, W_in: tl.constexpr, align_corners: tl.constexpr,
                                BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // (C * H_out * W_out)
    channel_idx = (pid % (C * H_out * W_out)) // (H_out * W_out)
    y_idx = (pid % (C * H_out * W_out)) % (H_out * W_out) // W_out
    x_idx = (pid % (C * H_out * W_out)) % (H_out * W_out) % W_out
    
    if batch_idx >= N:
        return
    
    # Load grid coordinates
    grid_x = tl.load(grid_ptr + batch_idx * H_out * W_out * 2 + y_idx * W_out * 2 + x_idx * 2 + 0)
    grid_y = tl.load(grid_ptr + batch_idx * H_out * W_out * 2 + y_idx * W_out * 2 + x_idx * 2 + 1)
    
    # Normalize grid coordinates to [0, W_in-1] or [0, H_in-1]
    if align_corners:
        x = (grid_x + 1.0) / 2.0 * (W_in - 1)
        y = (grid_y + 1.0) / 2.0 * (H_in - 1)
    else:
        x = (grid_x + 1.0) * (W_in - 1) / 2.0
        y = (grid_y + 1.0) * (H_in - 1) / 2.0
    
    # Clamp coordinates to valid range
    x = tl.maximum(0.0, tl.minimum(x, W_in - 1.0))
    y = tl.maximum(0.0, tl.minimum(y, H_in - 1.0))
    
    # Get integer coordinates
    x0 = tl.floor(x).to(tl.int32)
    y0 = tl.floor(y).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Clamp to image boundaries
    x1 = tl.minimum(x1, W_in - 1)
    y1 = tl.minimum(y1, H_in - 1)
    
    # Compute interpolation weights
    wx = x - x0
    wy = y - y0
    
    # Load pixel values
    input_ptr_base = input_ptr + batch_idx * C * H_in * W_in + channel_idx * H_in * W_in
    
    # Get four corner values
    val_00 = tl.load(input_ptr_base + y0 * W_in + x0, mask=(x0 < W_in) & (y0 < H_in), other=0.0)
    val_01 = tl.load(input_ptr_base + y0 * W_in + x1, mask=(x1 < W_in) & (y0 < H_in), other=0.0)
    val_10 = tl.load(input_ptr_base + y1 * W_in + x0, mask=(x0 < W_in) & (y1 < H_in), other=0.0)
    val_11 = tl.load(input_ptr_base + y1 * W_in + x1, mask=(x1 < W_in) & (y1 < H_in), other=0.0)
    
    # Bilinear interpolation
    val_0 = val_00 * (1 - wx) + val_01 * wx
    val_1 = val_10 * (1 - wx) + val_11 * wx
    result = val_0 * (1 - wy) + val_1 * wy
    
    # Store result
    tl.store(output_ptr + batch_idx * C * H_out * W_out + channel_idx * H_out * W_out + y_idx * W_out + x_idx, result)

def grid_sample_with_affine(input: torch.Tensor, theta: torch.Tensor, size: torch.Size, mode: str = 'bilinear', padding_mode: str = 'zeros', align_corners: bool = False) -> torch.Tensor:
    # Validate inputs
    assert input.dim() == 4, "input must be a 4D tensor (N, C, H, W)"
    assert theta.dim() == 3, "theta must be a 3D tensor (N, 2, 3)"
    assert theta.size(0) == input.size(0), "batch size mismatch between input and theta"
    assert theta.size(1) == 2 and theta.size(2) == 3, "theta must have shape (N, 2, 3)"
    assert len(size) == 4, "size must be a 4D size (N, C, H, W)"
    
    N, C, H_in, W_in = input.shape
    _, _, H_out, W_out = size
    
    # Generate grid using affine transformation
    grid = torch.empty(N, H_out, W_out, 2, dtype=torch.float32, device=input.device)
    
    # Create a temporary tensor for grid coordinates
    grid_flat = grid.view(N * H_out * W_out, 2)
    
    # Launch affine grid kernel
    block = 256
    grid_size = triton.cdiv(N * H_out * W_out, block)
    _affine_grid_kernel[grid_size](theta, grid_flat, N, H_out, W_out, BLOCK=block)
    
    # Apply grid sampling
    output = torch.empty(N, C, H_out, W_out, dtype=input.dtype, device=input.device)
    
    if mode == 'bilinear':
        # Launch grid sample kernel
        block = 256
        grid_size = triton.cdiv(N * C * H_out * W_out, block)
        _grid_sample_bilinear_kernel[grid_size](
            input, grid, output, 
            N, C, H_out, W_out, H_in, W_in, align_corners, 
            BLOCK=block
        )
    else:
        # For other modes, fall back to PyTorch implementation
        # This is a simplified version - in practice, you'd implement other modes
        output = torch.nn.functional.grid_sample(
            input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners
        )
    
    return output
