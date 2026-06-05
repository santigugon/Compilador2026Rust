import torch
import triton
import triton.language as tl

@triton.jit
def _combined_activation_kernel(
    input_ptr, weight1_ptr, weight2_ptr, bias_ptr, out_ptr,
    input_n, input_din, input_dout,
    input_stride_0, input_stride_1, input_stride_2,
    weight1_stride_0, weight1_stride_1,
    weight2_stride_0, weight2_stride_1, weight2_stride_2,
    bias_stride_0, bias_stride_1, bias_stride_2,
    out_stride_0, out_stride_1, out_stride_2,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // input_dout
    out_idx = pid % input_dout
    
    # Load input for this batch and output dimension
    input_offsets = batch_idx * input_stride_0 + tl.arange(0, input_din) * input_stride_1
    input_block = tl.load(input_ptr + input_offsets, mask=tl.arange(0, input_din) < input_din, other=0.0)
    
    # Matrix multiplication: input @ weight1
    weight1_offsets = tl.arange(0, input_din) * weight1_stride_0 + out_idx * weight1_stride_1
    weight1_block = tl.load(weight1_ptr + weight1_offsets, mask=tl.arange(0, input_din) < input_din, other=0.0)
    matmul_result = tl.sum(input_block * weight1_block, axis=0)
    
    # Apply sigmoid and tanh
    sigmoid_result = 1.0 / (1.0 + tl.exp(-matmul_result))
    tanh_result = 2.0 / (1.0 + tl.exp(-2.0 * matmul_result)) - 1.0
    
    # Element-wise multiplication with weight2
    weight2_offsets = batch_idx * weight2_stride_0 + out_idx * weight2_stride_1
    weight2_block = tl.load(weight2_ptr + weight2_offsets, mask=tl.arange(0, 1) < 1, other=0.0)
    elementwise_mult = sigmoid_result * weight2_block
    
    # Add bias
    bias_offsets = batch_idx * bias_stride_0 + out_idx * bias_stride_1
    bias_block = tl.load(bias_ptr + bias_offsets, mask=tl.arange(0, 1) < 1, other=0.0)
    final_result = elementwise_mult + bias_block
    
    # Store result
    out_offsets = batch_idx * out_stride_0 + out_idx * out_stride_1
    tl.store(out_ptr + out_offsets, final_result)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Validate dimensions
    batch_shape = input.shape[:-2]
    n, din = input.shape[-2], input.shape[-1]
    dout = weight1.shape[-1]
    
    # Ensure weight2 and bias are broadcastable
    if weight2.shape[-1] != dout:
        raise ValueError("weight2 must have the same last dimension as weight1")
    
    # Create output tensor
    if out is None:
        out = torch.empty(*batch_shape, n, dout, device=input.device, dtype=input.dtype)
    else:
        if out.shape != (*batch_shape, n, dout):
            raise ValueError("out tensor must have the correct shape")
    
    # Flatten batch dimensions for kernel launch
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    
    # Prepare strides
    input_stride_0 = input.stride(-3) if len(input.shape) >= 3 else 0
    input_stride_1 = input.stride(-2) if len(input.shape) >= 2 else 0
    input_stride_2 = input.stride(-1) if len(input.shape) >= 1 else 0
    
    weight1_stride_0 = weight1.stride(-2) if len(weight1.shape) >= 2 else 0
    weight1_stride_1 = weight1.stride(-1) if len(weight1.shape) >= 1 else 0
    
    weight2_stride_0 = weight2.stride(-2) if len(weight2.shape) >= 2 else 0
    weight2_stride_1 = weight2.stride(-1) if len(weight2.shape) >= 1 else 0
    weight2_stride_2 = weight2.stride(-1) if len(weight2.shape) >= 1 else 0
    
    bias_stride_0 = bias.stride(-2) if len(bias.shape) >= 2 else 0
    bias_stride_1 = bias.stride(-1) if len(bias.shape) >= 1 else 0
    bias_stride_2 = bias.stride(-1) if len(bias.shape) >= 1 else 0
    
    out_stride_0 = out.stride(-3) if len(out.shape) >= 3 else 0
    out_stride_1 = out.stride(-2) if len(out.shape) >= 2 else 0
    out_stride_2 = out.stride(-1) if len(out.shape) >= 1 else 0
    
    # Launch kernel
    block = 256
    grid = (batch_size * dout, 1, 1)
    
    _combined_activation_kernel[grid](
        input, weight1, weight2, bias, out,
        n, din, dout,
        input_stride_0, input_stride_1, input_stride_2,
        weight1_stride_0, weight1_stride_1,
        weight2_stride_0, weight2_stride_1, weight2_stride_2,
        bias_stride_0, bias_stride_1, bias_stride_2,
        out_stride_0, out_stride_1, out_stride_2,
        BLOCK=block
    )
    
    return out
