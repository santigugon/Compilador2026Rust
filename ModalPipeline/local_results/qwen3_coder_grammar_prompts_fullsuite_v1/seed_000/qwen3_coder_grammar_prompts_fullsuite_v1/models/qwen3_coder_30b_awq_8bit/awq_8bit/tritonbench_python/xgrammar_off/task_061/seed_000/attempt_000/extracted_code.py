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
    input_block = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
    
    # Compute matrix multiplication with weight1
    for k in range(0, input_din, BLOCK_SIZE):
        # Load weight1 column
        weight1_offsets = k + tl.arange(0, BLOCK_SIZE)
        weight1_mask = weight1_offsets < input_din
        
        # Load weight1 values
        weight1_block = tl.load(weight1_ptr + weight1_offsets, mask=weight1_mask, other=0.0)
        
        # Compute dot product
        acc += input_block * weight1_block
    
    # Convert to output size
    output_offsets = batch_id * seq_len * input_dout + seq_id * input_dout + tl.arange(0, input_dout)
    output_mask = output_offsets < (batch_id + 1) * seq_len * input_dout
    
    # Apply sigmoid and tanh
    output_block = tl.sigmoid(acc) * tl.tanh(acc)
    
    # Apply element-wise multiplication with weight2
    weight2_offsets = tl.arange(0, input_dout)
    weight2_mask = weight2_offsets < input_dout
    weight2_block = tl.load(weight2_ptr + weight2_offsets, mask=weight2_mask, other=0.0)
    output_block *= weight2_block
    
    # Add bias
    bias_offsets = tl.arange(0, input_dout)
    bias_mask = bias_offsets < input_dout
    bias_block = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
    output_block += bias_block
    
    # Store result
    tl.store(out_ptr + output_offsets, output_block, mask=output_mask)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Validate dimensions
    batch_dims = input.shape[:-2]
    seq_len, input_din = input.shape[-2:]
    output_dout = weight1.shape[-1]
    
    # Check compatibility
    assert input_din == weight1.shape[0], "Input dimension mismatch with weight1"
    assert weight2.shape[-1] == output_dout, "Weight2 dimension mismatch with output"
    assert bias.shape[-1] == output_dout, "Bias dimension mismatch with output"
    
    # Compute output shape
    output_shape = batch_dims + (seq_len, output_dout)
    
    # Create output tensor
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    else:
        assert out.shape == output_shape, "Output shape mismatch"
    
    # Handle batch dimensions
    batch_size = 1
    for dim in batch_dims:
        batch_size *= dim
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (batch_size, seq_len)
    
    # Flatten input for kernel processing
    input_flat = input.view(-1, input_din)
    out_flat = out.view(-1, output_dout)
    
    # Create a kernel that handles the full computation
    @triton.jit
    def _combined_activation_kernel_v2(
        input_ptr, weight1_ptr, weight2_ptr, bias_ptr, out_ptr,
        n_elements: tl.constexpr, 
        input_din: tl.constexpr,
        output_dout: tl.constexpr,
        batch_size: tl.constexpr,
        seq_len: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        batch_id = pid // seq_len
        seq_id = pid % seq_len
        
        # Calculate input offset
        input_offset = batch_id * seq_len * input_din + seq_id * input_din
        output_offset = batch_id * seq_len * output_dout + seq_id * output_dout
        
        # Process each output dimension
        for i in range(0, output_dout, BLOCK_SIZE):
            # Load input
            input_offsets = input_offset + tl.arange(0, BLOCK_SIZE)
            input_mask = input_offsets < (batch_id + 1) * seq_len * input_din
            
            # Load input block
            input_block = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
            
            # Compute output for this dimension
            acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            for k in range(0, input_din, BLOCK_SIZE):
                # Load weight1
                weight1_offsets = k + tl.arange(0, BLOCK_SIZE)
                weight1_mask = weight1_offsets < input_din
                
                # Load weight1 values
                weight1_block = tl.load(weight1_ptr + weight1_offsets, mask=weight1_mask, other=0.0)
                
                # Compute dot product
                acc += input_block * weight1_block
            
            # Apply activation functions
            output_block = tl.sigmoid(acc) * tl.tanh(acc)
            
            # Apply element-wise multiplication with weight2
            weight2_offsets = i + tl.arange(0, BLOCK_SIZE)
            weight2_mask = weight2_offsets < output_dout
            weight2_block = tl.load(weight2_ptr + weight2_offsets, mask=weight2_mask, other=0.0)
            output_block *= weight2_block
            
            # Add bias
            bias_offsets = i + tl.arange(0, BLOCK_SIZE)
            bias_mask = bias_offsets < output_dout
            bias_block = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
            output_block += bias_block
            
            # Store result
            output_offsets = output_offset + i + tl.arange(0, BLOCK_SIZE)
            output_mask = output_offsets < (batch_id + 1) * seq_len * output_dout
            tl.store(out_ptr + output_offsets, output_block, mask=output_mask)
    
    # Launch kernel
    grid_size = batch_size * seq_len
    _combined_activation_kernel_v2[grid_size](
        input, weight1, weight2, bias, out,
        input.numel(), input_din, output_dout, batch_size, seq_len, BLOCK_SIZE
    )
    
    return out
