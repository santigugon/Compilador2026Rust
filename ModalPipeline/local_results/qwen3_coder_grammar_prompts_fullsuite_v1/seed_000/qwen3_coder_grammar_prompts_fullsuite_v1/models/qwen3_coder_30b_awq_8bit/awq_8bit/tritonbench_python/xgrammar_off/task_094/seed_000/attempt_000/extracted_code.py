import torch
import triton
import triton.language as tl

@triton.jit
def dropout_sigmoid_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    n_features, n_out_features, p, training,
    stride_input_n, stride_input_f,
    stride_weight_o, stride_weight_f,
    stride_bias_o,
    stride_output_n, stride_output_o,
    BLOCK_SIZE: tl.constexpr
):
    # Get the block index
    block_idx = tl.program_id(0)
    
    # Calculate the starting position for this block
    start_pos = block_idx * BLOCK_SIZE
    
    # Load input data for this block
    input_block = tl.load(input_ptr + start_pos * stride_input_n, mask=start_pos < n_features)
    
    # Apply linear transformation
    output_block = tl.zeros((n_out_features,), dtype=tl.float32)
    
    for i in range(n_out_features):
        # Load weight for this output feature
        weight_row = tl.load(weight_ptr + i * stride_weight_o + start_pos * stride_weight_f, mask=start_pos < n_features)
        # Compute dot product
        output_block[i] = tl.sum(input_block * weight_row)
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_block = tl.load(bias_ptr + tl.arange(0, n_out_features) * stride_bias_o)
        output_block += bias_block
    
    # Apply sigmoid activation
    output_block = tl.sigmoid(output_block)
    
    # Apply dropout if training
    if training:
        # Generate random numbers for dropout
        rand_vals = tl.rand(0, n_out_features)
        # Zero out elements with probability p
        dropout_mask = rand_vals > p
        output_block = tl.where(dropout_mask, output_block, 0.0)
    
    # Store the result
    tl.store(output_ptr + start_pos * stride_output_n, output_block, mask=start_pos < n_features)

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Get dimensions
    n_features = input.numel()
    n_out_features = weight.shape[0]
    
    # Create output tensor
    if inplace:
        output = input
    else:
        output = torch.empty(input.shape[0], n_out_features, dtype=input.dtype, device=input.device)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid = (triton.cdiv(n_features, BLOCK_SIZE),)
    
    # Prepare pointers
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = output.data_ptr()
    
    # Launch kernel
    dropout_sigmoid_linear_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        n_features, n_out_features, p, training,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        bias.stride(0) if bias is not None else 0,
        output.stride(0), output.stride(1),
        BLOCK_SIZE
    )
    
    return output
