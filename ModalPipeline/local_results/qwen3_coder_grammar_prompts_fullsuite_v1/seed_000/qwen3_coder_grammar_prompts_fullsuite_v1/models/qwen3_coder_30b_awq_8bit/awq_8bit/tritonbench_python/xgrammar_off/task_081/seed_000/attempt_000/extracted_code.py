import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr, 
    input_height, input_width, 
    output_height, output_width,
    stride_h, stride_w,
    BLOCK_SIZE_H: tl.constexpr, 
    BLOCK_SIZE_W: tl.constexpr
):
    # Get block indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate output indices
    out_h = pid_h * BLOCK_SIZE_H
    out_w = pid_w * BLOCK_SIZE_W
    
    # Shared memory for reduction
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    # Loop over input regions
    for h_start in range(0, input_height, BLOCK_SIZE_H):
        for w_start in range(0, input_width, BLOCK_SIZE_W):
            # Calculate input region bounds
            h_end = min(h_start + BLOCK_SIZE_H, input_height)
            w_end = min(w_start + BLOCK_SIZE_W, input_width)
            
            # Calculate region size
            region_h = h_end - h_start
            region_w = w_end - w_start
            
            # Calculate average
            region_sum = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
            count = 0
            
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    val = tl.load(input_ptr + h * input_width + w)
                    region_sum += val
                    count += 1
            
            # Compute average
            if count > 0:
                avg_val = region_sum / count
                # Store in output
                if out_h < output_height and out_w < output_width:
                    tl.store(output_ptr + out_h * output_width + out_w, avg_val)
    
    # Apply sigmoid
    for h in range(BLOCK_SIZE_H):
        for w in range(BLOCK_SIZE_W):
            if out_h + h < output_height and out_w + w < output_width:
                val = tl.load(output_ptr + (out_h + h) * output_width + (out_w + w))
                sigmoid_val = 1.0 / (1.0 + tl.exp(-val))
                tl.store(output_ptr + (out_h + h) * output_width + (out_w + w), sigmoid_val)

def sigmoid_adaptive_avg_pool2d(input: torch.Tensor, output_size: torch.Size) -> torch.Tensor:
    # Ensure input is on GPU
    if input.device.type != 'cuda':
        input = input.cuda()
    
    # Handle output_size as int
    if isinstance(output_size, int):
        output_height = output_size
        output_width = output_size
    else:
        output_height, output_width = output_size
    
    # Get input dimensions
    input_height, input_width = input.shape[-2], input.shape[-1]
    
    # Create output tensor
    output = torch.empty(input.shape[:-2] + (output_height, output_width), dtype=torch.float32, device=input.device)
    
    # Define block size
    BLOCK_SIZE_H = 16
    BLOCK_SIZE_W = 16
    
    # Calculate grid size
    grid_h = (output_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (output_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    sigmoid_adaptive_avg_pool2d_kernel[(grid_h, grid_w)](
        input_ptr=input.data_ptr(),
        output_ptr=output.data_ptr(),
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        stride_h=1,
        stride_w=1,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W
    )
    
    return output
