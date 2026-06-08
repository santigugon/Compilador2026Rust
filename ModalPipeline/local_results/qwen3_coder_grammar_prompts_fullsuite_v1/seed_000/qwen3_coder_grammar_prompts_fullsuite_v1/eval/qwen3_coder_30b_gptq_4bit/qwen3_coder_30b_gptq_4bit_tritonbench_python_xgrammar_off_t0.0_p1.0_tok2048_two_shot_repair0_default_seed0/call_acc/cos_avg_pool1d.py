import torch
import triton
import triton.language as tl

@triton.jit
def _cos_kernel(x_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.cos(x)
    tl.store(out_ptr + offsets, y, mask=mask)

@triton.jit
def _avg_pool1d_kernel(
    x_ptr, out_ptr, 
    in_w: tl.constexpr, 
    out_w: tl.constexpr, 
    kernel_size: tl.constexpr, 
    stride: tl.constexpr, 
    padding: tl.constexpr,
    ceil_mode: tl.constexpr,
    count_include_pad: tl.constexpr,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    out_w_offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = out_w_offsets < out_w
    
    # Calculate output positions
    out_w_idx = out_w_offsets
    start_idx = out_w_idx * stride - padding
    
    # Compute pooling window boundaries
    window_start = start_idx
    window_end = window_start + kernel_size
    
    # Handle ceil mode
    if ceil_mode:
        # For ceil mode, we need to ensure we cover all input elements
        # This is a simplified approach for the kernel
        pass
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    count = tl.zeros((BLOCK,), dtype=tl.int32)
    
    # Loop over the kernel size
    for i in range(kernel_size):
        current_idx = window_start + i
        # Check if current index is within bounds
        valid_mask = (current_idx >= 0) & (current_idx < in_w)
        
        # Load input value
        x_val = tl.load(x_ptr + current_idx, mask=valid_mask & mask, other=0.0)
        
        # Accumulate
        accumulator = tl.where(valid_mask & mask, accumulator + x_val, accumulator)
        count = tl.where(valid_mask & mask, count + 1, count)
    
    # Compute average
    avg = tl.where(count > 0, accumulator / count, 0.0)
    
    # Store result
    tl.store(out_ptr + out_w_offsets, avg, mask=mask)

def cos_avg_pool1d(input: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> torch.Tensor:
    # Handle stride default
    if stride is None:
        stride = kernel_size
    
    # Get input dimensions
    minibatch, in_channels, iW = input.shape
    
    # Apply cosine function
    cos_input = torch.empty_like(input)
    n = input.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _cos_kernel[grid](input, cos_input, n, BLOCK=block)
    
    # Calculate output width
    if ceil_mode:
        out_w = (iW + 2 * padding - kernel_size) // stride + 1
        if (iW + 2 * padding - kernel_size) % stride != 0:
            out_w += 1
    else:
        out_w = (iW + 2 * padding - kernel_size) // stride + 1
    
    # Ensure out_w is not negative
    out_w = max(0, out_w)
    
    # Create output tensor
    out = torch.empty(minibatch, in_channels, out_w, dtype=input.dtype, device=input.device)
    
    # Apply average pooling
    if out_w > 0:
        block = 256
        grid = (triton.cdiv(out_w, block),)
        _avg_pool1d_kernel[grid](
            cos_input, out,
            iW, out_w, kernel_size, stride, padding,
            ceil_mode, count_include_pad, BLOCK=block
        )
    
    return out

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
