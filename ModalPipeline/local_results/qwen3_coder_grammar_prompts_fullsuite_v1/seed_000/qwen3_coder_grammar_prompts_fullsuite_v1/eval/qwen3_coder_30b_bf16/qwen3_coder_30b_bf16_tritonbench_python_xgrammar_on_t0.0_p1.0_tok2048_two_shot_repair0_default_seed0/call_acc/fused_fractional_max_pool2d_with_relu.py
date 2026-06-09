import torch
import triton
import triton.language as tl

def _get_kernel_size(kernel_size):
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    return kernel_size

def _get_output_size(input, output_size, output_ratio):
    if output_size is not None:
        return output_size
    if output_ratio is not None:
        h, w = input.shape[-2], input.shape[-1]
        return (int(h * output_ratio[0]), int(w * output_ratio[1]))
    return None

@triton.jit
def _relu_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _fractional_max_pool2d_kernel(
    input_ptr, output_ptr, indices_ptr,
    input_height, input_width,
    output_height, output_width,
    kernel_height, kernel_width,
    input_stride_0, input_stride_1,
    output_stride_0, output_stride_1,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    # Calculate output indices
    out_h_start = pid_h * BLOCK_H
    out_w_start = pid_w * BLOCK_W
    
    # Load input block
    input_block = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    indices_block = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.int32)
    
    for h in range(BLOCK_H):
        for w in range(BLOCK_W):
            if out_h_start + h < output_height and out_w_start + w < output_width:
                # Calculate input region
                h_start = (out_h_start + h) * kernel_height
                w_start = (out_w_start + w) * kernel_width
                
                # Find max in the region
                max_val = -float('inf')
                max_idx = -1
                
                for kh in range(kernel_height):
                    for kw in range(kernel_width):
                        h_in = h_start + kh
                        w_in = w_start + kw
                        if h_in < input_height and w_in < input_width:
                            val = tl.load(input_ptr + h_in * input_stride_0 + w_in * input_stride_1)
                            if val > max_val:
                                max_val = val
                                max_idx = h_in * input_stride_0 + w_in * input_stride_1
                
                input_block[h, w] = max_val
                indices_block[h, w] = max_idx
    
    # Store output
    for h in range(BLOCK_H):
        for w in range(BLOCK_W):
            if out_h_start + h < output_height and out_w_start + w < output_width:
                out_idx = (out_h_start + h) * output_stride_0 + (out_w_start + w) * output_stride_1
                tl.store(output_ptr + out_idx, input_block[h, w])
                if indices_ptr is not None:
                    tl.store(indices_ptr + out_idx, indices_block[h, w])

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    # Apply ReLU
    relu_input = torch.relu(input)
    
    # Get kernel size
    kernel_h, kernel_w = _get_kernel_size(kernel_size)
    
    # Get output size
    output_h, output_w = _get_output_size(relu_input, output_size, output_ratio)
    
    if output_h is None or output_w is None:
        # Default to full pooling
        output_h = (relu_input.shape[-2] + kernel_h - 1) // kernel_h
        output_w = (relu_input.shape[-1] + kernel_w - 1) // kernel_w
    
    # Create output tensor
    output = torch.empty(relu_input.shape[:-2] + (output_h, output_w), dtype=relu_input.dtype, device=relu_input.device)
    
    if return_indices:
        indices = torch.empty(relu_input.shape[:-2] + (output_h, output_w), dtype=torch.int32, device=relu_input.device)
    
    # Get strides
    input_stride_0 = relu_input.stride(-2)
    input_stride_1 = relu_input.stride(-1)
    output_stride_0 = output.stride(-2)
    output_stride_1 = output.stride(-1)
    
    # Launch kernel
    grid_h = triton.cdiv(output_h, 16)
    grid_w = triton.cdiv(output_w, 16)
    grid = (grid_h, grid_w)
    
    if return_indices:
        _fractional_max_pool2d_kernel[grid](
            relu_input, output, indices,
            relu_input.shape[-2], relu_input.shape[-1],
            output_h, output_w,
            kernel_h, kernel_w,
            input_stride_0, input_stride_1,
            output_stride_0, output_stride_1,
            BLOCK_H=16, BLOCK_W=16
        )
        return output, indices
    else:
        _fractional_max_pool2d_kernel[grid](
            relu_input, output, None,
            relu_input.shape[-2], relu_input.shape[-1],
            output_h, output_w,
            kernel_h, kernel_w,
            input_stride_0, input_stride_1,
            output_stride_0, output_stride_1,
            BLOCK_H=16, BLOCK_W=16
        )
        return output
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
#     relu_output = F.relu(input)
#     pooled_output = F.fractional_max_pool2d(relu_output, kernel_size=kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices)
#     return pooled_output

def test_fused_fractional_max_pool2d_with_relu():
    results = {}
    
    # Test case 1: Basic functionality with kernel_size and output_size
    input_tensor = torch.randn(1, 1, 8, 8, device='cuda')
    kernel_size = (2, 2)
    output_size = (4, 4)
    results["test_case_1"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size)
    
    # Test case 2: Using output_ratio instead of output_size
    output_ratio = (0.5, 0.5)
    results["test_case_2"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_ratio=output_ratio)
    
    # Test case 3: Return indices along with the pooled output
    results["test_case_3"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size, return_indices=True)
    
    # Test case 4: Larger kernel size
    kernel_size = (3, 3)
    results["test_case_4"] = fused_fractional_max_pool2d_with_relu(input_tensor, kernel_size, output_size=output_size)
    
    return results

test_results = test_fused_fractional_max_pool2d_with_relu()
