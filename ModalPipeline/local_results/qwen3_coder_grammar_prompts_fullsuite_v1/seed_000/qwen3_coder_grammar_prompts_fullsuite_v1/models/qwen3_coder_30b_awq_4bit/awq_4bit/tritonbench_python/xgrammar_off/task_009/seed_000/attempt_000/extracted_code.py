import torch
import triton
import triton.language as tl

@triton.jit
def grid_sample_kernel(
    input_ptr, grid_ptr, output_ptr,
    input_shape, grid_shape,
    mode, padding_mode, align_corners,
    BLOCK_SIZE=128
):
    # Get thread index
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Handle 4D and 5D cases
    batch_size = input_shape[0]
    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]
    
    # Grid dimensions
    grid_height = grid_shape[1]
    grid_width = grid_shape[2]
    
    # Output dimensions
    out_height = grid_height
    out_width = grid_width
    
    # Calculate output indices
    batch_idx = idx // (out_height * out_width)
    out_y = (idx % (out_height * out_width)) // out_width
    out_x = idx % out_width
    
    # Check bounds
    if batch_idx >= batch_size or out_y >= out_height or out_x >= out_width:
        return
    
    # Get grid coordinates
    grid_y = tl.load(grid_ptr + batch_idx * grid_height * grid_width * 2 + 
                     out_y * grid_width * 2 + out_x * 2)
    grid_x = tl.load(grid_ptr + batch_idx * grid_height * grid_width * 2 + 
                     out_y * grid_width * 2 + out_x * 2 + 1)
    
    # Normalize coordinates based on align_corners
    if align_corners:
        grid_y = (grid_y + 1) / 2.0 * (height - 1)
        grid_x = (grid_x + 1) / 2.0 * (width - 1)
    else:
        grid_y = (grid_y + 1) / 2.0 * height
        grid_x = (grid_x + 1) / 2.0 * width
    
    # Handle padding modes
    if padding_mode == 'zeros':
        if grid_y < 0 or grid_y >= height or grid_x < 0 or grid_x >= width:
            tl.store(output_ptr + batch_idx * out_height * out_width + 
                     out_y * out_width + out_x, 0.0)
            return
    
    # Bilinear interpolation
    if mode == 'bilinear':
        # Get integer coordinates
        y0 = tl.floor(grid_y).to(tl.int32)
        x0 = tl.floor(grid_x).to(tl.int32)
        y1 = y0 + 1
        x1 = x0 + 1
        
        # Clamp coordinates
        y0 = tl.clamp(y0, 0, height - 1)
        x0 = tl.clamp(x0, 0, width - 1)
        y1 = tl.clamp(y1, 0, height - 1)
        x1 = tl.clamp(x1, 0, width - 1)
        
        # Calculate weights
        wy = grid_y - y0
        wx = grid_x - x0
        
        # Get input values
        val00 = tl.load(input_ptr + batch_idx * height * width + 
                        y0 * width + x0)
        val01 = tl.load(input_ptr + batch_idx * height * width + 
                        y0 * width + x1)
        val10 = tl.load(input_ptr + batch_idx * height * width + 
                        y1 * width + x0)
        val11 = tl.load(input_ptr + batch_idx * height * width + 
                        y1 * width + x1)
        
        # Interpolate
        val0 = val00 * (1 - wx) + val01 * wx
        val1 = val10 * (1 - wx) + val11 * wx
        output_val = val0 * (1 - wy) + val1 * wy
        
        tl.store(output_ptr + batch_idx * out_height * out_width + 
                 out_y * out_width + out_x, output_val)
    else:  # nearest neighbor
        y = tl.round(grid_y).to(tl.int32)
        x = tl.round(grid_x).to(tl.int32)
        y = tl.clamp(y, 0, height - 1)
        x = tl.clamp(x, 0, width - 1)
        output_val = tl.load(input_ptr + batch_idx * height * width + 
                            y * width + x)
        tl.store(output_ptr + batch_idx * out_height * out_width + 
                 out_y * out_width + out_x, output_val)

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    # Validate inputs
    if input.dim() not in [4, 5]:
        raise ValueError("Input must be 4D or 5D")
    if grid.dim() not in [4, 5]:
        raise ValueError("Grid must be 4D or 5D")
    
    # Prepare output tensor
    batch_size = input.shape[0]
    channels = input.shape[1]
    out_height = grid.shape[1]
    out_width = grid.shape[2]
    
    # Create output tensor
    if input.dim() == 4:
        output = torch.empty(batch_size, channels, out_height, out_width, 
                           device=input.device, dtype=input.dtype)
    else:  # 5D
        out_depth = grid.shape[3]
        output = torch.empty(batch_size, channels, out_height, out_width, out_depth,
                           device=input.device, dtype=input.dtype)
    
    # Prepare input and grid pointers
    input_ptr = input.contiguous().data_ptr()
    grid_ptr = grid.contiguous().data_ptr()
    output_ptr = output.data_ptr()
    
    # Calculate grid size
    if input.dim() == 4:
        total_elements = batch_size * out_height * out_width
    else:
        total_elements = batch_size * out_height * out_width * out_depth
    
    # Launch kernel
    grid_size = (total_elements + 128 - 1) // 128
    grid_sample_kernel[grid_size](
        input_ptr, grid_ptr, output_ptr,
        input.shape, grid.shape,
        mode, padding_mode, align_corners
    )
    
    return output
