import torch
import triton
import triton.language as tl

@triton.jit
def _combined_activation_kernel(
    input_ptr, weight1_ptr, weight2_ptr, bias_ptr, out_ptr,
    input_n, input_din, input_dout,
    batch_size, seq_len,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Calculate offsets for batch and sequence dimensions
    batch_offset = batch_id * seq_len * input_din
    seq_offset = seq_id * input_din
    
    # Load input for this batch and sequence
    input_offsets = batch_offset + seq_offset + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < (batch_id + 1) * seq_len * input_din
    
    # Perform matrix multiplication: input @ weight1
    # We'll compute this in chunks to handle large dimensions
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load input
    input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Compute matrix multiplication with weight1
    for i in range(0, input_din, BLOCK_SIZE):
        weight1_offsets = i * input_dout + tl.arange(0, BLOCK_SIZE)
        weight1_mask = weight1_offsets < input_din * input_dout
        
        # Load weight1
        weight1_vals = tl.load(weight1_ptr + weight1_offsets, mask=weight1_mask, other=0.0)
        
        # Compute dot product
        for j in range(BLOCK_SIZE):
            if i + j < input_din:
                acc += input_vals[j] * weight1_vals[j * input_dout]
    
    # Apply sigmoid and tanh
    sigmoid_vals = 1.0 / (1.0 + tl.exp(-acc))
    tanh_vals = 2.0 / (1.0 + tl.exp(-2.0 * acc)) - 1.0
    
    # Element-wise multiplication with weight2
    # weight2 is broadcastable, so we'll compute it per element
    weight2_offsets = tl.arange(0, BLOCK_SIZE)
    weight2_mask = weight2_offsets < input_dout
    weight2_vals = tl.load(weight2_ptr + weight2_offsets, mask=weight2_mask, other=0.0)
    
    # Combine sigmoid and tanh
    combined = sigmoid_vals * tanh_vals
    
    # Element-wise multiplication with weight2
    result = combined * weight2_vals
    
    # Add bias
    bias_offsets = tl.arange(0, BLOCK_SIZE)
    bias_mask = bias_offsets < input_dout
    bias_vals = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
    result += bias_vals
    
    # Store result
    out_offsets = batch_id * seq_len * input_dout + seq_id * input_dout + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < (batch_id + 1) * seq_len * input_dout
    tl.store(out_ptr + out_offsets, result, mask=out_mask)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Validate input dimensions
    assert input.dim() >= 2, "input must have at least 2 dimensions"
    assert weight1.dim() == 2, "weight1 must be 2-dimensional"
    assert weight1.shape[1] == input.shape[-1], "weight1's second dimension must match input's last dimension"
    
    # Get batch dimensions
    batch_shape = input.shape[:-2]
    seq_len = input.shape[-2]
    input_din = input.shape[-1]
    input_dout = weight1.shape[1]
    
    # Validate weight2 and bias shapes for broadcasting
    assert weight2.shape[-1] == input_dout, "weight2's last dimension must match weight1's second dimension"
    assert bias.shape[-1] == input_dout, "bias's last dimension must match weight1's second dimension"
    
    # Prepare output tensor
    if out is None:
        out_shape = batch_shape + (seq_len, input_dout)
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == batch_shape + (seq_len, input_dout), "out tensor has incorrect shape"
    
    # Launch kernel
    block_size = 256
    batch_size = 1
    for dim in batch_shape:
        batch_size *= dim
    
    grid = (batch_size, seq_len)
    
    # Flatten input for kernel processing
    input_flat = input.view(-1, input_din)
    out_flat = out.view(-1, input_dout)
    
    # Launch kernel
    _combined_activation_kernel[grid](
        input_flat, weight1, weight2, bias, out_flat,
        input_din, input_din, input_dout,
        batch_size, seq_len,
        BLOCK_SIZE=block_size
    )
    
    return out
