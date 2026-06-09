import torch
import triton
import triton.language as tl

def _ceil_div(a, b):
    return (a + b - 1) // b

def _get_output_size(input_size, kernel_size, stride, padding, ceil_mode):
    if ceil_mode:
        return _ceil_div(input_size + 2 * padding - kernel_size, stride) + 1
    else:
        return (input_size + 2 * padding - kernel_size) // stride + 1

@triton.jit
def _cos_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.cos(x)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _avg_pool1d_kernel(input_ptr, output_ptr, input_size: tl.constexpr, output_size: tl.constexpr,
                       kernel_size: tl.constexpr, stride: tl.constexpr, padding: tl.constexpr,
                       count_include_pad: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    output_offset = pid * BLOCK
    
    # Each thread processes one output element
    if output_offset >= output_size:
        return
    
    # Calculate the start and end indices in the input for this output element
    start_input = output_offset * stride - padding
    end_input = start_input + kernel_size
    
    # Initialize sum and count
    sum_val = 0.0
    count = 0
    
    # Iterate through the kernel window
    for i in range(kernel_size):
        input_idx = start_input + i
        # Check if input index is valid
        if input_idx >= 0 and input_idx < input_size:
            # Load input value
            input_val = tl.load(input_ptr + input_idx, mask=True, other=0.0)
            sum_val += input_val
            count += 1
        elif count_include_pad:
            # If padding is included, we still count the padding (which is 0)
            count += 1
    
    # Compute average
    if count > 0:
        avg = sum_val / count
    else:
        avg = 0.0
    
    # Store result
    tl.store(output_ptr + output_offset, avg, mask=output_offset < output_size)


def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    # Handle default stride
    if stride is None:
        stride = kernel_size
    
    # Get input dimensions
    batch_size, channels, input_size = input.shape
    
    # Compute output size
    output_size = _get_output_size(input_size, kernel_size, stride, padding, ceil_mode)
    
    # Apply cosine to input
    cos_input = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _cos_kernel[grid](input, cos_input, n, BLOCK=block)
    
    # Apply average pooling
    output = torch.empty(batch_size, channels, output_size, dtype=input.dtype, device=input.device)
    
    # Process each batch and channel
    for b in range(batch_size):
        for c in range(channels):
            # Flatten the input and output for this batch and channel
            input_flat = cos_input[b, c].contiguous().view(-1)
            output_flat = output[b, c].contiguous().view(-1)
            
            # Launch kernel for this flattened tensor
            n_output = output_flat.numel()
            block = 256
            grid = (triton.cdiv(n_output, block),)
            _avg_pool1d_kernel[grid](
                input_flat, output_flat,
                input_size, output_size,
                kernel_size, stride, padding,
                count_include_pad,
                BLOCK=block
            )
    
    return output
##################################################################################################################################################



import torch
import torch.nn.functional as F

# def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int=None, padding: int=0, ceil_mode: bool=False, count_include_pad: bool=True) -> torch.Tensor:
#     cos_input = torch.cos(input)
#     return F.avg_pool1d(cos_input, kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad)

def test_cos_avg_pool1d():
    results = {}

    # Test case 1: Basic functionality with default parameters
    input_tensor_1 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_1'] = cos_avg_pool1d(input_tensor_1, kernel_size=2)

    # Test case 2: Custom stride
    input_tensor_2 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_2'] = cos_avg_pool1d(input_tensor_2, kernel_size=2, stride=1)

    # Test case 3: With padding
    input_tensor_3 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_3'] = cos_avg_pool1d(input_tensor_3, kernel_size=2, padding=1)

    # Test case 4: Using ceil_mode
    input_tensor_4 = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]], device='cuda')
    results['test_case_4'] = cos_avg_pool1d(input_tensor_4, kernel_size=2, ceil_mode=True)

    return results

test_results = test_cos_avg_pool1d()
