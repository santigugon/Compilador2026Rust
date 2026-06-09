import torch
import triton
import triton.language as tl

@triton.jit
def grid_sample_kernel(
    input_ptr, grid_ptr, output_ptr,
    input_shape0, input_shape1, input_shape2, input_shape3,
    grid_shape0, grid_shape1, grid_shape2,
    output_shape0, output_shape1, output_shape2,
    mode, padding_mode, align_corners,
    BLOCK_SIZE: tl.constexpr
):
    # Get thread indices
    batch_idx = tl.program_id(0)
    y_idx = tl.program_id(1)
    x_idx = tl.program_id(2)
    
    # Calculate output index
    output_idx = batch_idx * output_shape1 * output_shape2 + y_idx * output_shape2 + x_idx
    
    # Get grid coordinates
    grid_x = tl.load(grid_ptr + batch_idx * grid_shape1 * grid_shape2 + y_idx * grid_shape2 + x_idx)
    grid_y = tl.load(grid_ptr + batch_idx * grid_shape1 * grid_shape2 + y_idx * grid_shape2 + x_idx + output_shape1 * output_shape2)
    
    # Normalize grid coordinates
    if align_corners:
        grid_x = grid_x * 0.5 + 0.5
        grid_y = grid_y * 0.5 + 0.5
    else:
        grid_x = (grid_x + 1.0) * 0.5
        grid_y = (grid_y + 1.0) * 0.5
    
    # Clamp coordinates to valid range
    grid_x = tl.clamp(grid_x, 0.0, 1.0)
    grid_y = tl.clamp(grid_y, 0.0, 1.0)
    
    # Convert to input coordinates
    input_x = grid_x * (input_shape3 - 1)
    input_y = grid_y * (input_shape2 - 1)
    
    # Handle padding mode
    if padding_mode == 0:  # zeros
        if input_x < 0 or input_x >= input_shape3 or input_y < 0 or input_y >= input_shape2:
            tl.store(output_ptr + output_idx, 0.0)
            return
    
    # Bilinear interpolation
    if mode == 1:  # bilinear
        x0 = tl.floor(input_x).to(tl.int32)
        y0 = tl.floor(input_y).to(tl.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Clamp to valid range
        x0 = tl.clamp(x0, 0, input_shape3 - 1)
        x1 = tl.clamp(x1, 0, input_shape3 - 1)
        y0 = tl.clamp(y0, 0, input_shape2 - 1)
        y1 = tl.clamp(y1, 0, input_shape2 - 1)
        
        # Get weights
        wx = input_x - x0
        wy = input_y - y0
        
        # Sample four corners
        val00 = tl.load(input_ptr + batch_idx * input_shape1 * input_shape2 * input_shape3 + 
                       y0 * input_shape3 + x0)
        val01 = tl.load(input_ptr + batch_idx * input_shape1 * input_shape2 * input_shape3 + 
                       y0 * input_shape3 + x1)
        val10 = tl.load(input_ptr + batch_idx * input_shape1 * input_shape2 * input_shape3 + 
                       y1 * input_shape3 + x0)
        val11 = tl.load(input_ptr + batch_idx * input_shape1 * input_shape2 * input_shape3 + 
                       y1 * input_shape3 + x1)
        
        # Interpolate
        val0 = val00 * (1 - wx) + val01 * wx
        val1 = val10 * (1 - wx) + val11 * wx
        result = val0 * (1 - wy) + val1 * wy
        
        tl.store(output_ptr + output_idx, result)
    else:  # nearest
        x = tl.round(input_x).to(tl.int32)
        y = tl.round(input_y).to(tl.int32)
        x = tl.clamp(x, 0, input_shape3 - 1)
        y = tl.clamp(y, 0, input_shape2 - 1)
        val = tl.load(input_ptr + batch_idx * input_shape1 * input_shape2 * input_shape3 + 
                     y * input_shape3 + x)
        tl.store(output_ptr + output_idx, val)

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    # Validate inputs
    assert input.dim() in [4, 5], "Input must be 4D or 5D"
    assert grid.dim() == 4, "Grid must be 4D"
    assert input.shape[0] == grid.shape[0], "Batch dimensions must match"
    assert grid.shape[3] == 2, "Grid must have 2 channels"
    
    # Determine mode and padding mode
    mode_enum = 1 if mode == 'bilinear' else 0
    padding_enum = 0 if padding_mode == 'zeros' else 1
    
    # Get shapes
    input_shape = input.shape
    grid_shape = grid.shape
    output_shape = [input_shape[0], grid_shape[1], grid_shape[2]]
    
    # Create output tensor
    output = torch.empty(output_shape[0], output_shape[1], output_shape[2], 
                        dtype=input.dtype, device=input.device)
    
    # Launch kernel
    if input.dim() == 4:
        # 4D case: [N, C, H, W]
        grid_sample_kernel[(output_shape[0], output_shape[1], output_shape[2]), 
                          (1, 1, 1)](
            input.data_ptr(), grid.data_ptr(), output.data_ptr(),
            input_shape[0], input_shape[1], input_shape[2], input_shape[3],
            grid_shape[0], grid_shape[1], grid_shape[2],
            output_shape[0], output_shape[1], output_shape[2],
            mode_enum, padding_enum, align_corners,
            BLOCK_SIZE=1024
        )
    else:
        # 5D case: [N, C, D, H, W]
        output_shape = [input_shape[0], input_shape[1], grid_shape[1], grid_shape[2]]
        output = torch.empty(output_shape[0], output_shape[1], output_shape[2], output_shape[3], 
                            dtype=input.dtype, device=input.device)
        grid_sample_kernel[(output_shape[0], output_shape[2], output_shape[3]), 
                          (1, 1, 1)](
            input.data_ptr(), grid.data_ptr(), output.data_ptr(),
            input_shape[0], input_shape[1], input_shape[2], input_shape[3],
            grid_shape[0], grid_shape[1], grid_shape[2],
            output_shape[0], output_shape[2], output_shape[3],
            mode_enum, padding_enum, align_corners,
            BLOCK_SIZE=1024
        )
    
    return output
