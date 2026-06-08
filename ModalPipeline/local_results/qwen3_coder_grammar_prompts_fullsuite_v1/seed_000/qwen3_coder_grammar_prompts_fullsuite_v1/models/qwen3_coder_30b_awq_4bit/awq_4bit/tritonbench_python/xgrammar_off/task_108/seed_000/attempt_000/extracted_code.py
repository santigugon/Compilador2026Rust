import torch
import triton
import triton.language as tl

@triton.jit
def affine_grid_kernel(
    theta_ptr, 
    grid_ptr, 
    N, 
    H_out, 
    W_out, 
    H_in, 
    W_in,
    BLOCK_SIZE: tl.constexpr
):
    # Compute global thread index
    batch_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    w_idx = tl.program_id(2)
    
    if batch_idx >= N or h_idx >= H_out or w_idx >= W_out:
        return
    
    # Load theta for this batch
    theta = tl.load(theta_ptr + batch_idx * 6)
    theta_1 = tl.load(theta_ptr + batch_idx * 6 + 1)
    theta_2 = tl.load(theta_ptr + batch_idx * 6 + 2)
    theta_3 = tl.load(theta_ptr + batch_idx * 6 + 3)
    theta_4 = tl.load(theta_ptr + batch_idx * 6 + 4)
    theta_5 = tl.load(theta_ptr + batch_idx * 6 + 5)
    
    # Compute grid coordinates
    h = h_idx
    w = w_idx
    
    # Normalize coordinates to [-1, 1] range
    h_norm = (h / (H_out - 1)) * 2 - 1
    w_norm = (w / (W_out - 1)) * 2 - 1
    
    # Apply affine transformation
    x = theta_0 * w_norm + theta_1 * h_norm + theta_2
    y = theta_3 * w_norm + theta_4 * h_norm + theta_5
    
    # Store grid values
    grid_idx = batch_idx * H_out * W_out * 2 + h_idx * W_out * 2 + w_idx * 2
    tl.store(grid_ptr + grid_idx, x)
    tl.store(grid_ptr + grid_idx + 1, y)

@triton.jit
def grid_sample_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    N,
    C,
    H_out,
    W_out,
    H_in,
    W_in,
    mode,
    padding_mode,
    align_corners,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)
    
    if batch_idx >= N or channel_idx >= C or h_idx >= H_out or w_idx >= W_out:
        return
    
    # Load grid coordinates
    grid_idx = batch_idx * H_out * W_out * 2 + h_idx * W_out * 2 + w_idx * 2
    x = tl.load(grid_ptr + grid_idx)
    y = tl.load(grid_ptr + grid_idx + 1)
    
    # Normalize coordinates based on align_corners
    if align_corners:
        x = (x + 1) / 2 * (W_in - 1)
        y = (y + 1) / 2 * (H_in - 1)
    else:
        x = (x + 1) / 2 * W_in
        y = (y + 1) / 2 * H_in
    
    # Handle padding modes
    if padding_mode == "zeros":
        if x < 0 or x >= W_in or y < 0 or y >= H_in:
            tl.store(output_ptr + batch_idx * C * H_out * W_out + channel_idx * H_out * W_out + h_idx * W_out + w_idx, 0.0)
            return
    elif padding_mode == "border":
        x = tl.clamp(x, 0, W_in - 1)
        y = tl.clamp(y, 0, H_in - 1)
    elif padding_mode == "reflection":
        x = tl.where(x < 0, -x, x)
        y = tl.where(y < 0, -y, y)
        x = x % (2 * W_in - 2)
        y = y % (2 * H_in - 2)
        x = tl.where(x >= W_in, 2 * W_in - 2 - x, x)
        y = tl.where(y >= H_in, 2 * H_in - 2 - y, y)
    
    # Interpolation modes
    if mode == "nearest":
        x_int = tl.cast(x + 0.5, tl.int32)
        y_int = tl.cast(y + 0.5, tl.int32)
        x_int = tl.clamp(x_int, 0, W_in - 1)
        y_int = tl.clamp(y_int, 0, H_in - 1)
        input_idx = batch_idx * C * H_in * W_in + channel_idx * H_in * W_in + y_int * W_in + x_int
        tl.store(output_ptr + batch_idx * C * H_out * W_out + channel_idx * H_out * W_out + h_idx * W_out + w_idx, tl.load(input_ptr + input_idx))
    elif mode == "bilinear":
        x0 = tl.cast(x, tl.int32)
        y0 = tl.cast(y, tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Clamp coordinates
        x0 = tl.clamp(x0, 0, W_in - 1)
        y0 = tl.clamp(y0, 0, H_in - 1)
        x1 = tl.clamp(x1, 0, W_in - 1)
        y1 = tl.clamp(y1, 0, H_in - 1)
        
        # Compute weights
        wx = x - x0
        wy = y - y0
        
        # Sample four corners
        idx00 = batch_idx * C * H_in * W_in + channel_idx * H_in * W_in + y0 * W_in + x0
        idx01 = batch_idx * C * H_in * W_in + channel_idx * H_in * W_in + y0 * W_in + x1
        idx10 = batch_idx * C * H_in * W_in + channel_idx * H_in * W_in + y1 * W_in + x0
        idx11 = batch_idx * C * H_in * W_in + channel_idx * H_in * W_in + y1 * W_in + x1
        
        val00 = tl.load(input_ptr + idx00)
        val01 = tl.load(input_ptr + idx01)
        val10 = tl.load(input_ptr + idx10)
        val11 = tl.load(input_ptr + idx11)
        
        # Bilinear interpolation
        val0 = val00 * (1 - wx) + val01 * wx
        val1 = val10 * (1 - wx) + val11 * wx
        result = val0 * (1 - wy) + val1 * wy
        
        tl.store(output_ptr + batch_idx * C * H_out * W_out + channel_idx * H_out * W_out + h_idx * W_out + w_idx, result)
    else:
        # Default to bilinear for unsupported modes
        x0 = tl.cast(x, tl.int32)
        y0 = tl.cast(y, tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Clamp coordinates
        x0 = tl.clamp(x0, 0, W_in - 1)
        y0 = tl.clamp(y0, 0, H_in - 1)
        x1 = tl.clamp(x1, 0, W_in - 1)
        y1 = tl.clamp(y1, 0, H_in - 1)
        
        # Compute weights
        wx = x - x0
        wy = y - y0
        
        # Sample four corners
        idx00 = batch_idx * C * H_in * W_in + channel_idx * H_in * W_in + y0 * W_in + x0
        idx01 = batch_idx * C * H_in * W_in + channel_idx * H_in * W_in + y0 * W_in + x1
        idx10 = batch_idx * C * H_in * W_in + channel_idx * H_in * W_in + y1 * W_in + x0
        idx11 = batch_idx * C * H_in * W_in + channel_idx * H_in * W_in + y1 * W_in + x1
        
        val00 = tl.load(input_ptr + idx00)
        val01 = tl.load(input_ptr + idx01)
        val10 = tl.load(input_ptr + idx10)
        val11 = tl.load(input_ptr + idx11)
        
        # Bilinear interpolation
        val0 = val00 * (1 - wx) + val01 * wx
        val1 = val10 * (1 - wx) + val11 * wx
        result = val0 * (1 - wy) + val1 * wy
        
        tl.store(output_ptr + batch_idx * C * H_out * W_out + channel_idx * H_out * W_out + h_idx * W_out + w_idx, result)

def grid_sample_with_affine(input: torch.Tensor, theta: torch.Tensor, size:
