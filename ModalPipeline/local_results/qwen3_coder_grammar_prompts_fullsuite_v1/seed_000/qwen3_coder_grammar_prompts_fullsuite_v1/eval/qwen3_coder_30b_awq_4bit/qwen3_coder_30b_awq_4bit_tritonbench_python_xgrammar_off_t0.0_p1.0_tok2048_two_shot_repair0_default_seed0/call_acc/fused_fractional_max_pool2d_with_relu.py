import torch
import triton
import triton.language as tl

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
    input_ptr, 
    output_ptr, 
    indices_ptr,
    input_height, 
    input_width, 
    output_height, 
    output_width,
    kernel_height, 
    kernel_width,
    BLOCK_H: tl.constexpr, 
    BLOCK_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    if pid_h >= output_height or pid_w >= output_width:
        return
    
    # Calculate the start position in the input tensor
    start_h = pid_h * kernel_height
    start_w = pid_w * kernel_width
    
    # Load the kernel_size for this block
    max_val = -float('inf')
    max_idx = 0
    
    # Loop through the kernel window to find max
    for h in range(kernel_height):
        for w in range(kernel_width):
            input_h = start_h + h
            input_w = start_w + w
            if input_h < input_height and input_w < input_width:
                input_offset = input_h * input_width + input_w
                val = tl.load(input_ptr + input_offset)
                if val > max_val:
                    max_val = val
                    max_idx = input_offset
    
    # Store the result
    output_offset = pid_h * output_width + pid_w
    tl.store(output_ptr + output_offset, max_val)
    
    if indices_ptr is not None:
        tl.store(indices_ptr + output_offset, max_idx)

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False):
    # Handle kernel_size
    if isinstance(kernel_size, int):
        kernel_height = kernel_size
        kernel_width = kernel_size
    else:
        kernel_height, kernel_width = kernel_size
    
    # Apply ReLU first
    input_re = torch.relu(input)
    
    # Get input dimensions
    input_height, input_width = input_re.shape[-2], input_re.shape[-1]
    
    # Calculate output size if not provided
    if output_size is not None:
        output_height, output_width = output_size
    elif output_ratio is not None:
        output_height = int(input_height * output_ratio[0])
        output_width = int(input_width * output_ratio[1])
    else:
        # Default to kernel size
        output_height = input_height // kernel_height
        output_width = input_width // kernel_width
    
    # Create output tensor
    if len(input.shape) == 4:
        out = torch.empty(input.shape[0], input.shape[1], output_height, output_width, device=input.device, dtype=input.dtype)
        if return_indices:
            indices = torch.empty(input.shape[0], input.shape[1], output_height, output_width, device=input.device, dtype=torch.long)
    else:
        out = torch.empty(output_height, output_width, device=input.device, dtype=input.dtype)
        if return_indices:
            indices = torch.empty(output_height, output_width, device=input.device, dtype=torch.long)
    
    # Handle batch dimension
    batch_size = 1
    if len(input.shape) == 4:
        batch_size = input.shape[0]
    
    # Launch kernel
    if return_indices:
        if len(input.shape) == 4:
            for b in range(batch_size):
                _fractional_max_pool2d_kernel[(output_height, output_width)](
                    input_re[b], 
                    out[b], 
                    indices[b],
                    input_height, 
                    input_width, 
                    output_height, 
                    output_width,
                    kernel_height, 
                    kernel_width,
                    BLOCK_H=16, 
                    BLOCK_W=16,
                    BLOCK_SIZE=256
                )
        else:
            _fractional_max_pool2d_kernel[(output_height, output_width)](
                input_re, 
                out, 
                indices,
                input_height, 
                input_width, 
                output_height, 
                output_width,
                kernel_height, 
                kernel_width,
                BLOCK_H=16, 
                BLOCK_W=16,
                BLOCK_SIZE=256
            )
        return out, indices
    else:
        if len(input.shape) == 4:
            for b in range(batch_size):
                _fractional_max_pool2d_kernel[(output_height, output_width)](
                    input_re[b], 
                    out[b], 
                    None,
                    input_height, 
                    input_width, 
                    output_height, 
                    output_width,
                    kernel_height, 
                    kernel_width,
                    BLOCK_H=16, 
                    BLOCK_W=16,
                    BLOCK_SIZE=256
                )
        else:
            _fractional_max_pool2d_kernel[(output_height, output_width)](
                input_re, 
                out, 
                None,
                input_height, 
                input_width, 
                output_height, 
                output_width,
                kernel_height, 
                kernel_width,
                BLOCK_H=16, 
                BLOCK_W=16,
                BLOCK_SIZE=256
            )
        return out

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
