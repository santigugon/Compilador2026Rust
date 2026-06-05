import torch
import triton
import triton.language as tl

@triton.jit
def _combined_activation_kernel(
    input_ptr, weight1_ptr, weight2_ptr, bias_ptr, out_ptr,
    input_n, input_d_in, input_d_out,
    batch_size, seq_len,
    input_stride_0, input_stride_1, input_stride_2,
    weight1_stride_0, weight1_stride_1,
    weight2_stride_0, weight2_stride_1, weight2_stride_2,
    bias_stride_0, bias_stride_1, bias_stride_2,
    out_stride_0, out_stride_1, out_stride_2,
    BLOCK_SIZE: tl.constexpr
):
    # Get the batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Calculate the base offsets for the current batch and sequence
    input_base = batch_idx * input_stride_0 + seq_idx * input_stride_1
    weight2_base = batch_idx * weight2_stride_0 + seq_idx * weight2_stride_1
    bias_base = batch_idx * bias_stride_0 + seq_idx * bias_stride_1
    out_base = batch_idx * out_stride_0 + seq_idx * out_stride_1
    
    # Loop over the output dimension
    for i in range(0, input_d_out, BLOCK_SIZE):
        # Create masks for the current block
        mask = (i + tl.arange(0, BLOCK_SIZE)) < input_d_out
        
        # Load weight1 for this block
        weight1_offsets = tl.arange(0, BLOCK_SIZE) + i
        weight1_block = tl.load(weight1_ptr + weight1_offsets, mask=mask, other=0.0)
        
        # Compute matrix multiplication for this block
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(0, input_d_in, BLOCK_SIZE):
            # Create masks for the current block
            mask_j = (j + tl.arange(0, BLOCK_SIZE)) < input_d_in
            
            # Load input for this block
            input_offsets = input_base + (j + tl.arange(0, BLOCK_SIZE)) * input_stride_2
            input_block = tl.load(input_ptr + input_offsets, mask=mask_j, other=0.0)
            
            # Compute dot product
            acc += tl.sum(input_block[:, None] * weight1_block[None, :], axis=0)
        
        # Apply sigmoid and tanh
        sigmoid_val = 1.0 / (1.0 + tl.exp(-acc))
        tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * acc)) - 1.0
        
        # Element-wise multiplication with weight2
        weight2_offsets = weight2_base + (i + tl.arange(0, BLOCK_SIZE)) * weight2_stride_2
        weight2_block = tl.load(weight2_ptr + weight2_offsets, mask=mask, other=0.0)
        result = sigmoid_val * tanh_val * weight2_block
        
        # Add bias
        bias_offsets = bias_base + (i + tl.arange(0, BLOCK_SIZE)) * bias_stride_2
        bias_block = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)
        result += bias_block
        
        # Store the result
        out_offsets = out_base + (i + tl.arange(0, BLOCK_SIZE)) * out_stride_2
        tl.store(out_ptr + out_offsets, result, mask=mask)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Validate input dimensions
    assert input.dim() >= 2, "input must have at least 2 dimensions"
    assert weight1.dim() == 2, "weight1 must be 2-dimensional"
    assert weight1.shape[1] == input.shape[-1], "weight1's second dimension must match input's last dimension"
    
    # Get batch dimensions
    batch_dims = input.shape[:-2]
    seq_len = input.shape[-2]
    input_d_in = input.shape[-1]
    input_d_out = weight1.shape[1]
    
    # Compute output shape
    output_shape = batch_dims + (seq_len, input_d_out)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape, "out tensor must have the correct shape"
        assert out.dtype == input.dtype, "out tensor must have the same dtype as input"
        assert out.device == input.device, "out tensor must be on the same device as input"
    
    # Flatten batch dimensions for kernel launch
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Get strides
    input_stride_0 = input.stride(-3) if len(input.shape) >= 3 else 0
    input_stride_1 = input.stride(-2) if len(input.shape) >= 2 else 0
    input_stride_2 = input.stride(-1) if len(input.shape) >= 1 else 0
    
    weight1_stride_0 = weight1.stride(0) if weight1.dim() >= 2 else 0
    weight1_stride_1 = weight1.stride(1) if weight1.dim() >= 2 else 0
    
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
    grid = (batch_size, seq_len)
    
    _combined_activation_kernel[grid](
        input, weight1, weight2, bias, out,
        input_d_in, input_d_in, input_d_out,
        batch_size, seq_len,
        input_stride_0, input_stride_1, input_stride_2,
        weight1_stride_0, weight1_stride_1,
        weight2_stride_0, weight2_stride_1, weight2_stride_2,
        bias_stride_0, bias_stride_1, bias_stride_2,
        out_stride_0, out_stride_1, out_stride_2,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
