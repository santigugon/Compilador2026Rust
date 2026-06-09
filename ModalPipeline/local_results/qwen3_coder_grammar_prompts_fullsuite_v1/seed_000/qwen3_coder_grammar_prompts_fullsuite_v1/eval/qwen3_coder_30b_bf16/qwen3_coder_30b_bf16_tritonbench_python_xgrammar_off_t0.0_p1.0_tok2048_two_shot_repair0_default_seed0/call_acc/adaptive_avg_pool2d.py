import torch
import triton
import triton.language as tl

@triton.jit
def _adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    input_strides_0, input_strides_1, input_strides_2, input_strides_3,
    output_strides_0, output_strides_1, output_strides_2, output_strides_3,
    input_shape_0, input_shape_1, input_shape_2, input_shape_3,
    output_shape_0, output_shape_1, output_shape_2, output_shape_3,
    H_out, W_out, H_in, W_in,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    # Get the block indices
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate the start indices for this block
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    
    # Calculate the end indices for this block
    h_end = min(h_start + BLOCK_H, H_out)
    w_end = min(w_start + BLOCK_W, W_out)
    
    # Loop over the output dimensions
    for h in range(h_start, h_end):
        for w in range(w_start, w_end):
            # Calculate the input region boundaries
            h_start_in = (h * H_in) // H_out
            h_end_in = ((h + 1) * H_in + H_out - 1) // H_out
            w_start_in = (w * W_in) // W_out
            w_end_in = ((w + 1) * W_in + W_out - 1) // W_out
            
            # Calculate the sum of values in the region
            sum_val = 0.0
            count = 0
            
            for ih in range(h_start_in, h_end_in):
                for iw in range(w_start_in, w_end_in):
                    # Calculate the input index
                    input_idx = (
                        ih * input_strides_2 + 
                        iw * input_strides_3
                    )
                    # Load the value
                    val = tl.load(input_ptr + input_idx, mask=True)
                    sum_val += val
                    count += 1
            
            # Calculate the average
            avg_val = sum_val / count if count > 0 else 0.0
            
            # Calculate the output index
            output_idx = (
                h * output_strides_2 + 
                w * output_strides_3
            )
            # Store the result
            tl.store(output_ptr + output_idx, avg_val)

def adaptive_avg_pool2d(input, output_size):
    # Handle the case where input is (C, H_in, W_in) or (N, C, H_in, W_in)
    if input.dim() == 3:
        # (C, H_in, W_in)
        N, C, H_in, W_in = 1, input.shape[0], input.shape[1], input.shape[2]
        input = input.unsqueeze(0)  # Make it (1, C, H_in, W_in)
    else:
        # (N, C, H_in, W_in)
        N, C, H_in, W_in = input.shape
    
    # Handle output_size
    if isinstance(output_size, int):
        H_out, W_out = output_size, output_size
    elif isinstance(output_size, tuple) and len(output_size) == 2:
        H_out, W_out = output_size
    else:
        raise ValueError("output_size must be an int or a tuple of two ints")
    
    # Create output tensor
    output = torch.empty(N, C, H_out, W_out, device=input.device, dtype=input.dtype)
    
    # Define block size
    BLOCK_H = 16
    BLOCK_W = 16
    
    # Calculate grid size
    grid_h = triton.cdiv(H_out, BLOCK_H)
    grid_w = triton.cdiv(W_out, BLOCK_W)
    grid = (grid_h, grid_w)
    
    # Get strides and shapes
    input_strides = input.stride()
    output_strides = output.stride()
    
    # Launch kernel
    _adaptive_avg_pool2d_kernel[grid](
        input, output,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3],
        N, C, H_in, W_in,
        N, C, H_out, W_out,
        H_out, W_out, H_in, W_in,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    
    # Remove the extra dimension if needed
    if input.dim() == 3:
        output = output.squeeze(0)
    
    return output

##################################################################################################################################################



import torch
from adaptive_avg_pool2d import adaptive_avg_pool2d

def test_adaptive_avg_pool2d():
    results = {}
    
    # Test case 1: input shape (N, C, H_in, W_in), output_size as integer
    input1 = torch.randn(1, 64, 8, 9).cuda()
    output1 = adaptive_avg_pool2d(input1, 5)
    results["test_case_1"] = output1
    
    # Test case 2: input shape (N, C, H_in, W_in), output_size as tuple
    input2 = torch.randn(1, 64, 8, 9).cuda()
    output2 = adaptive_avg_pool2d(input2, (5, 7))
    results["test_case_2"] = output2
    
    # Test case 3: input shape (C, H_in, W_in), output_size as integer
    input3 = torch.randn(64, 8, 9).cuda()
    output3 = adaptive_avg_pool2d(input3, 5)
    results["test_case_3"] = output3
    
    # Test case 4: input shape (C, H_in, W_in), output_size as tuple
    input4 = torch.randn(64, 8, 9).cuda()
    output4 = adaptive_avg_pool2d(input4, (5, 7))
    results["test_case_4"] = output4
    
    return results

test_results = test_adaptive_avg_pool2d()
