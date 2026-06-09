import torch
import triton
import triton.language as tl

def tanh_linear(input, weight, bias=None):
    # Check if input is a tensor
    if not torch.is_tensor(input):
        input = torch.tensor(input)
    if not torch.is_tensor(weight):
        weight = torch.tensor(weight)
    if bias is not None and not torch.is_tensor(bias):
        bias = torch.tensor(bias)

    # Get dimensions
    input_shape = input.shape
    in_features = input_shape[-1]
    out_features = weight.shape[0]
    
    # Reshape input to 2D for matrix multiplication
    input_2d = input.view(-1, in_features)
    batch_size = input_2d.shape[0]
    
    # Output tensor
    out = torch.empty(batch_size, out_features, device=input.device, dtype=input.dtype)
    
    # Handle bias
    if bias is not None:
        bias_2d = bias.view(1, out_features)
    else:
        bias_2d = None
    
    # Launch kernel
    _tanh_linear_kernel[batch_size](
        input_2d, weight, bias_2d, out,
        input_2d.stride(0), weight.stride(0), weight.stride(1),
        out.stride(0),
        in_features, out_features,
        BLOCK_SIZE=32
    )
    
    # Reshape output back to original shape
    return out.view(input_shape[:-1] + (out_features,))

@triton.jit
def _tanh_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, out_ptr,
    input_stride_0, weight_stride_0, weight_stride_1,
    out_stride_0,
    in_features: tl.constexpr,
    out_features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Load input
    input_row = tl.load(input_ptr + pid * input_stride_0 + tl.arange(0, in_features),
                        mask=tl.arange(0, in_features) < in_features,
                        other=0.0)
    
    # Compute linear transformation
    output_row = tl.zeros((out_features,), dtype=tl.float32)
    
    # Loop over weight matrix
    for i in range(0, in_features, BLOCK_SIZE):
        # Load weight block
        weight_block = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE) + i,
                              mask=tl.arange(0, BLOCK_SIZE) + i < in_features,
                              other=0.0)
        
        # Compute dot product
        for j in range(out_features):
            output_row[j] += tl.sum(input_row * weight_block)
    
    # Add bias if present
    if bias_ptr is not None:
        bias_row = tl.load(bias_ptr + tl.arange(0, out_features),
                           mask=tl.arange(0, out_features) < out_features,
                           other=0.0)
        output_row += bias_row
    
    # Apply tanh activation
    output_row = 2.0 / (1.0 + tl.exp(-2.0 * output_row)) - 1.0
    
    # Store result
    tl.store(out_ptr + pid * out_stride_0 + tl.arange(0, out_features),
             output_row,
             mask=tl.arange(0, out_features) < out_features)