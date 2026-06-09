import torch
import triton
import triton.language as tl

@triton.jit
def _combined_activation_kernel(
    input_ptr, weight1_ptr, weight2_ptr, bias_ptr, out_ptr,
    batch_size: tl.constexpr, n: tl.constexpr, d_in: tl.constexpr, d_out: tl.constexpr,
    input_stride_0: tl.constexpr, input_stride_1: tl.constexpr, input_stride_2: tl.constexpr,
    weight1_stride_0: tl.constexpr, weight1_stride_1: tl.constexpr,
    weight2_stride_0: tl.constexpr, weight2_stride_1: tl.constexpr, weight2_stride_2: tl.constexpr,
    bias_stride_0: tl.constexpr, bias_stride_1: tl.constexpr, bias_stride_2: tl.constexpr,
    out_stride_0: tl.constexpr, out_stride_1: tl.constexpr, out_stride_2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    out_idx = tl.program_id(1)
    
    # Load input
    input_offsets = batch_idx * input_stride_0 + out_idx * input_stride_1 + tl.arange(0, BLOCK_SIZE)
    input_block = tl.load(input_ptr + input_offsets, mask=input_offsets < batch_size * n * d_in, other=0.0)
    
    # Matrix multiplication with weight1
    # We'll compute the output for one batch and one output dimension
    # For simplicity, we'll compute the full matrix multiplication in a single kernel
    # This is a simplified approach - in practice, you might want to use a more optimized approach
    
    # Compute intermediate result: input @ weight1
    intermediate = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for k in range(d_in):
        input_val = tl.load(input_ptr + batch_idx * input_stride_0 + out_idx * input_stride_1 + k * input_stride_2, mask=k < d_in, other=0.0)
        weight_val = tl.load(weight1_ptr + k * weight1_stride_0 + tl.arange(0, BLOCK_SIZE) * weight1_stride_1, mask=tl.arange(0, BLOCK_SIZE) < d_out, other=0.0)
        intermediate += input_val * weight_val
    
    # Apply sigmoid and tanh
    sigmoid_result = 1.0 / (1.0 + tl.exp(-intermediate))
    tanh_result = 2.0 / (1.0 + tl.exp(-2.0 * intermediate)) - 1.0
    
    # Element-wise multiplication with weight2
    weight2_offsets = batch_idx * weight2_stride_0 + out_idx * weight2_stride_1 + tl.arange(0, BLOCK_SIZE) * weight2_stride_2
    weight2_block = tl.load(weight2_ptr + weight2_offsets, mask=weight2_offsets < batch_size * n * d_out, other=0.0)
    elementwise_result = sigmoid_result * weight2_block
    
    # Add bias
    bias_offsets = batch_idx * bias_stride_0 + out_idx * bias_stride_1 + tl.arange(0, BLOCK_SIZE) * bias_stride_2
    bias_block = tl.load(bias_ptr + bias_offsets, mask=bias_offsets < batch_size * n * d_out, other=0.0)
    final_result = elementwise_result + bias_block
    
    # Store result
    out_offsets = batch_idx * out_stride_0 + out_idx * out_stride_1 + tl.arange(0, BLOCK_SIZE) * out_stride_2
    tl.store(out_ptr + out_offsets, final_result, mask=out_offsets < batch_size * n * d_out)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Validate input dimensions
    batch_shape = input.shape[:-2]
    n, d_in = input.shape[-2], input.shape[-1]
    d_out = weight1.shape[-1]
    
    # Check compatibility
    assert weight1.shape == (d_in, d_out), f"Weight1 shape mismatch: expected ({d_in}, {d_out}), got {weight1.shape}"
    
    # Compute output shape
    output_shape = batch_shape + (n, d_out)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape, f"Output shape mismatch: expected {output_shape}, got {out.shape}"
        
    # Flatten batch dimensions for easier processing
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    
    # Get strides
    input_stride_0 = input.stride(-3) if len(input.shape) >= 3 else 0
    input_stride_1 = input.stride(-2) if len(input.shape) >= 2 else 0
    input_stride_2 = input.stride(-1) if len(input.shape) >= 1 else 0
    
    weight1_stride_0 = weight1.stride(0)
    weight1_stride_1 = weight1.stride(1)
    
    weight2_stride_0 = weight2.stride(-3) if len(weight2.shape) >= 3 else 0
    weight2_stride_1 = weight2.stride(-2) if len(weight2.shape) >= 2 else 0
    weight2_stride_2 = weight2.stride(-1) if len(weight2.shape) >= 1 else 0
    
    bias_stride_0 = bias.stride(-3) if len(bias.shape) >= 3 else 0
    bias_stride_1 = bias.stride(-2) if len(bias.shape) >= 2 else 0
    bias_stride_2 = bias.stride(-1) if len(bias.shape) >= 1 else 0
    
    out_stride_0 = out.stride(-3) if len(out.shape) >= 3 else 0
    out_stride_1 = out.stride(-2) if len(out.shape) >= 2 else 0
    out_stride_2 = out.stride(-1) if len(out.shape) >= 1 else 0
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (batch_size, triton.cdiv(n, BLOCK_SIZE))
    
    _combined_activation_kernel[grid](
        input, weight1, weight2, bias, out,
        batch_size, n, d_in, d_out,
        input_stride_0, input_stride_1, input_stride_2,
        weight1_stride_0, weight1_stride_1,
        weight2_stride_0, weight2_stride_1, weight2_stride_2,
        bias_stride_0, bias_stride_1, bias_stride_2,
        out_stride_0, out_stride_1, out_stride_2,
        BLOCK_SIZE
    )
    
    return out