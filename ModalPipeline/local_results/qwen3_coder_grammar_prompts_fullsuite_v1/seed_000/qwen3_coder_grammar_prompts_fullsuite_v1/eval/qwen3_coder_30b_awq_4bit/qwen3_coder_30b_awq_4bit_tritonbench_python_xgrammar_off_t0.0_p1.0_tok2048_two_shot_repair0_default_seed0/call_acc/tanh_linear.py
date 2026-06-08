import torch
import triton
import triton.language as tl

@triton.jit
def _tanh_linear_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, 
                        in_features: tl.constexpr, out_features: tl.constexpr, 
                        batch_size: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    batch_id = pid // out_features
    out_feature_id = pid % out_features
    
    if batch_id >= batch_size:
        return
    
    # Compute linear transformation
    acc = 0.0
    for i in range(0, in_features, BLOCK):
        input_offsets = batch_id * in_features + i + tl.arange(0, BLOCK)
        weight_offsets = out_feature_id * in_features + i + tl.arange(0, BLOCK)
        
        input_mask = (i + tl.arange(0, BLOCK)) < in_features
        weight_mask = (i + tl.arange(0, BLOCK)) < in_features
        
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        weight_vals = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
        
        acc += tl.sum(input_vals * weight_vals)
    
    # Add bias if present
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + out_feature_id)
        acc += bias_val
    
    # Apply tanh activation
    tanh_val = 2.0 / (1.0 + tl.exp(-2.0 * acc)) - 1.0
    
    # Store result
    output_offsets = batch_id * out_features + out_feature_id
    tl.store(output_ptr + output_offsets, tanh_val)

def tanh_linear(input, weight, bias=None):
    # Handle the case where input has multiple dimensions
    batch_shape = input.shape[:-1]
    in_features = input.shape[-1]
    out_features = weight.shape[0]
    
    # Flatten input to 2D for easier processing
    input_flat = input.view(-1, in_features)
    batch_size = input_flat.shape[0]
    
    # Create output tensor
    out = torch.empty(batch_size, out_features, device=input.device, dtype=input.dtype)
    
    # Determine block size
    BLOCK = 256
    
    # Launch kernel
    num_blocks = batch_size * out_features
    grid = (triton.cdiv(num_blocks, BLOCK),)
    
    # Handle bias being None
    bias_ptr = bias if bias is not None else None
    
    # Launch kernel with appropriate parameters
    _tanh_linear_kernel[grid](
        input_flat, 
        weight, 
        bias_ptr, 
        out, 
        in_features, 
        out_features, 
        batch_size, 
        BLOCK=BLOCK
    )
    
    # Reshape output to match input shape
    out = out.view(*batch_shape, out_features)
    
    return out

##################################################################################################################################################



import torch
from tanh_linear import tanh_linear

def test_tanh_linear():
    results = {}

    # Test case 1: input, weight, and bias on GPU
    input1 = torch.randn(5, 3, device='cuda')
    weight1 = torch.randn(4, 3, device='cuda')
    bias1 = torch.randn(4, device='cuda')
    result1 = tanh_linear(input1, weight1, bias1)
    results["test_case_1"] = result1

    # Test case 2: input and weight on GPU, bias is None
    input2 = torch.randn(5, 3, device='cuda')
    weight2 = torch.randn(4, 3, device='cuda')
    result2 = tanh_linear(input2, weight2)
    results["test_case_2"] = result2

    # Test case 3: input and weight on GPU, bias on GPU
    input3 = torch.randn(2, 3, device='cuda')
    weight3 = torch.randn(2, 3, device='cuda')
    bias3 = torch.randn(2, device='cuda')
    result3 = tanh_linear(input3, weight3, bias3)
    results["test_case_3"] = result3

    # Test case 4: input, weight, and bias on GPU with different dimensions
    input4 = torch.randn(3, 2, device='cuda')
    weight4 = torch.randn(2, 2, device='cuda')
    bias4 = torch.randn(2, device='cuda')
    result4 = tanh_linear(input4, weight4, bias4)
    results["test_case_4"] = result4

    return results

test_results = test_tanh_linear()
