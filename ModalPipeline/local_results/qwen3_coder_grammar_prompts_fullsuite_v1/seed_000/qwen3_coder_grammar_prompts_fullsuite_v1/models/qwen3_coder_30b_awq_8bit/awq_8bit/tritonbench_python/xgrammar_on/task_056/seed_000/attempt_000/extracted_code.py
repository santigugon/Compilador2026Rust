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
    out_h = pid_h * BLOCK_H
    out_w = pid_w * BLOCK_W
    
    # Load input block
    input_block = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    indices_block = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.int32)
    
    # Pooling
    for kh in range(kernel_height):
        for kw in range(kernel_width):
            h = out_h + kh
            w = out_w + kw
            if h < input_height and w < input_width:
                input_val = tl.load(input_ptr + h * input_stride_0 + w * input_stride_1)
                # Update max
                mask = input_val > input_block
                input_block = tl.where(mask, input_val, input_block)
                # Update indices
                indices_block = tl.where(mask, (h * input_width + w).to(tl.int32), indices_block)
    
    # Store output
    for i in range(BLOCK_H):
        for j in range(BLOCK_W):
            if out_h + i < output_height and out_w + j < output_width:
                tl.store(output_ptr + (out_h + i) * output_stride_0 + (out_w + j) * output_stride_1,
                        input_block[i, j])
                if indices_ptr is not None:
                    tl.store(indices_ptr + (out_h + i) * output_stride_0 + (out_w + j) * output_stride_1,
                            indices_block[i, j])

def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    # Apply ReLU
    input_relu = torch.relu(input)
    
    # Get kernel size
    kernel_h, kernel_w = _get_kernel_size(kernel_size)
    
    # Get output size
    output_h, output_w = _get_output_size(input, output_size, output_ratio)
    
    if output_h is None or output_w is None:
        # Default to 1x1 output if not specified
        output_h, output_w = 1, 1
    
    # Create output tensor
    out = torch.empty(input.shape[0], input.shape[1], output_h, output_w, device=input.device, dtype=input.dtype)
    
    if return_indices:
        indices = torch.empty(input.shape[0], input.shape[1], output_h, output_w, device=input.device, dtype=torch.int32)
    else:
        indices = None
    
    # Launch kernel
    if output_h > 0 and output_w > 0:
        block_h, block_w = 16, 16
        grid_h = triton.cdiv(output_h, block_h)
        grid_w = triton.cdiv(output_w, block_w)
        grid = (grid_h, grid_w)
        
        _fractional_max_pool2d_kernel[grid](
            input_relu, out, indices,
            input.shape[2], input.shape[3],
            output_h, output_w,
            kernel_h, kernel_w,
            input.stride(2), input.stride(3),
            out.stride(2), out.stride(3),
            BLOCK_H=block_h,
            BLOCK_W=block_w
        )
    
    if return_indices:
        return out, indices
    return out