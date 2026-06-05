import torch
import triton
import triton.language as tl

@triton.jit
def _adaptive_avg_pool2d_kernel(
    input_ptr, output_ptr,
    input_stride_0, input_stride_1, input_stride_2, input_stride_3,
    output_stride_0, output_stride_1, output_stride_2, output_stride_3,
    n_elements, H_in, W_in, H_out, W_out,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)
    
    # Calculate which output element this block is responsible for
    output_idx = block_start
    
    while output_idx < block_end:
        # Calculate output indices
        out_n = output_idx // (H_out * W_out)
        out_c = (output_idx // W_out) % H_out
        out_h = output_idx % W_out
        
        # Calculate input region bounds
        h_start = (out_h * H_in) // H_out
        h_end = ((out_h + 1) * H_in + H_out - 1) // H_out
        w_start = (out_c * W_in) // W_out
        w_end = ((out_c + 1) * W_in + W_out - 1) // W_out
        
        # Calculate average
        sum_val = 0.0
        count = 0
        
        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                input_idx = out_n * input_stride_0 + out_c * input_stride_1 + h * input_stride_2 + w * input_stride_3
                sum_val += tl.load(input_ptr + input_idx, mask=True)
                count += 1
        
        # Store average
        output_idx = out_n * output_stride_0 + out_c * output_stride_1 + out_h * output_stride_2 + out_c * output_stride_3
        if count > 0:
            avg_val = sum_val / count
            tl.store(output_ptr + output_idx, avg_val, mask=True)
        
        output_idx += 1

def adaptive_avg_pool2d(input, output_size):
    # Handle input shape and output size
    if len(input.shape) == 3:
        # (C, H_in, W_in)
        N, C, H_in, W_in = 1, input.shape[0], input.shape[1], input.shape[2]
        input_reshaped = input.unsqueeze(0)
    else:
        # (N, C, H_in, W_in)
        N, C, H_in, W_in = input.shape
        input_reshaped = input
    
    # Handle output size
    if isinstance(output_size, int):
        H_out, W_out = output_size, output_size
    elif isinstance(output_size, tuple) and len(output_size) == 2:
        H_out, W_out = output_size
    else:
        raise ValueError("output_size must be an int or a tuple of two ints")
    
    # Handle None values
    if H_out is None:
        H_out = H_in
    if W_out is None:
        W_out = W_in
    
    # Create output tensor
    output = torch.empty(N, C, H_out, W_out, device=input.device, dtype=input.dtype)
    
    # Calculate number of elements in output
    n_elements = N * C * H_out * W_out
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Get strides
    input_stride_0 = input_reshaped.stride(0)
    input_stride_1 = input_reshaped.stride(1)
    input_stride_2 = input_reshaped.stride(2)
    input_stride_3 = input_reshaped.stride(3)
    
    output_stride_0 = output.stride(0)
    output_stride_1 = output.stride(1)
    output_stride_2 = output.stride(2)
    output_stride_3 = output.stride(3)
    
    _adaptive_avg_pool2d_kernel[grid](
        input_reshaped, output,
        input_stride_0, input_stride_1, input_stride_2, input_stride_3,
        output_stride_0, output_stride_1, output_stride_2, output_stride_3,
        n_elements, H_in, W_in, H_out, W_out,
        BLOCK_SIZE
    )
    
    # Return appropriate shape
    if len(input.shape) == 3:
        return output.squeeze(0)
    else:
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
